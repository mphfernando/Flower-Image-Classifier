###
import argparse
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights
from PIL import Image
import numpy as np
import json
from train import Classifier  # Import the custom classifier


def load_checkpoint(filepath):
    """Load a trained model from a checkpoint file safely."""
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)

    # Print checkpoint keys for debugging
    print("Checkpoint keys:", checkpoint.keys())

    # Load pre-trained VGG16 model
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)

    # Freeze feature extractor parameters
    for param in model.features.parameters():
        param.requires_grad = False

    # Retrieve classifier parameters from checkpoint
    input_size = checkpoint.get('input_size', 25088)  # Default for VGG16
    hidden_layer_1 = checkpoint.get('hidden_layer_1_units', 512)
    hidden_layer_2 = checkpoint.get('hidden_layer_2_units', 256)
    output_size = checkpoint.get('output_size', 102)  # Default for 102 flower classes

    # Rebuild classifier
    model.classifier = Classifier(input_size, hidden_layer_1, hidden_layer_2, output_size)

    # Load trained weights
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("ERROR: 'state_dict' missing! Ensure correct checkpoint format.")

    # Load class-to-index mapping
    model.class_to_idx = checkpoint.get('class_to_idx', {})

    print("Checkpoint loaded successfully!")
    return model


def process_image(image_path):
    """Preprocess an image for model inference."""
    image = Image.open(image_path).convert("RGB")  # Convert to RGB

    # Resize to maintain aspect ratio
    image = image.resize((256, 256))

    # Center crop to 224x224
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert image to numpy array and normalize
    np_image = np.array(image) / 255.0  # Scale pixel values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions (H, W, C) -> (C, H, W)
    np_image = np_image.transpose((2, 0, 1))

    return torch.tensor(np_image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


def predict(image_path, model, topk=5, device="cpu"):
    """Predict the top K classes for a given image."""
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Preprocess the image
    image = process_image(image_path).to(device)

    # Forward pass through the model
    with torch.no_grad():
        output = model(image)
        probs, indices = torch.topk(F.softmax(output, dim=1), topk)

    # Convert to numpy arrays
    probs = probs.cpu().numpy().flatten()
    indices = indices.cpu().numpy().flatten()

    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in indices]

    return probs, classes


def main():
    """Main function for parsing arguments and running inference."""
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained network.")
    parser.add_argument("imagepath", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()

    # Select device (GPU if available and specified)
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    # Load model
    model = load_checkpoint(args.checkpoint)
    model.to(device)

    # Run prediction
    probs, classes = predict(args.imagepath, model, args.top_k, device)

    # Convert class indices to real names if mapping is provided
    if args.category_names:
        try:
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)
            classes = [cat_to_name.get(str(cls), cls) for cls in classes]
        except Exception as e:
            print(f"Error loading category names: {e}")

    # Print results
    print("\n🔹 Predicted Classes:", classes)
    print("🔹 Probabilities:", probs)


if __name__ == '__main__':
    main()
