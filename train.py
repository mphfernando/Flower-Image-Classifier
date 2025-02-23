###
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Function to load data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32, shuffle=False),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False)
    }

    return dataloaders, image_datasets

# Define the classifier class
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_layer_1_units, hidden_layer_2_units, output_size=102):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_1_units)
        self.fc2 = nn.Linear(hidden_layer_1_units, hidden_layer_2_units)
        self.fc3 = nn.Linear(hidden_layer_2_units, output_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Function to build the model
def build_model(arch, hidden_layer_1_units, hidden_layer_2_units):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError("Only 'vgg16' and 'vgg13' architectures are supported.")

    input_size = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False  # Freeze feature extraction layers

    model.classifier = Classifier(input_size, hidden_layer_1_units, hidden_layer_2_units)
    return model

# Function to train the model
def train_model(model, dataloaders, epochs=5, learning_rate=1e-3, use_gpu=True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print_every = 10
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item()

                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {val_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()

    return model

# Function to save checkpoint
def save_checkpoint(model, optimizer, image_datasets, save_path="flower_classifier.pth"):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'model_name': 'vgg16',
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved successfully at {save_path}!")



# Main execution
def main():
    parser = argparse.ArgumentParser(description="Train a neural network on flower data.")
    parser.add_argument("data_directory", type=str, help="Path to the dataset")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_layer_1_units", type=int, default=512, help="Hidden layer 1 units")
    parser.add_argument("--hidden_layer_2_units", type=int, default=256, help="Hidden layer 2 units")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    dataloaders, image_datasets = load_data(args.data_directory)
    model = build_model(args.arch, args.hidden_layer_1_units, args.hidden_layer_2_units)
    trained_model = train_model(model, dataloaders, args.epochs, args.learning_rate, args.gpu)
    save_checkpoint(trained_model, optim.Adam(model.classifier.parameters(), lr=args.learning_rate), image_datasets, f"{args.save_dir}/flower_classifier.pth")

if __name__ == '__main__':
    main()
