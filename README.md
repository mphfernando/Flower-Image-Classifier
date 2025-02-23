# 🌸 Flower Image Classifier using Deep Learning

This project implements a **deep learning-based image classifier** to identify different flower species using a **pretrained VGG16 model**. The classifier is trained on a dataset of flower images, leveraging **transfer learning** and **PyTorch** to fine-tune the model.

## 📌 Features
- **Data Augmentation & Normalization** using `torchvision.transforms`
- **Pretrained Model (VGG16)** with a custom feedforward classifier
- **Training with GPU support** for faster computations
- **Top-K Class Predictions** for accurate classification
- **Matplotlib Visualization** for image predictions
- **Command-line Interface** (`train.py` & `predict.py`)

## 🏗 Project Structure
```
├── train.py          # Script to train the model
├── predict.py        # Script to predict flower species from an image
├── Image_Classifier_Project.ipynb  # Jupyter Notebook with training & evaluation
├── cat_to_name.json  # JSON file mapping class indices to flower names
├── flower_classifier.pth    # Saved model checkpoint
├── README.md         # Project documentation
```

## 📂 Dataset
The dataset consists of images of various flower species, organized into three folders:
- **Train** (`/flowers/train/`)
- **Validation** (`/flowers/valid/`)
- **Test** (`/flowers/test/`)

Each folder contains images categorized into subdirectories by flower species.

## 🔧 Installation & Setup
To run this project, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/flower-classifier.git
   cd flower-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision matplotlib numpy pandas
   ```

3. **Download and prepare the dataset**
   Place the dataset inside the `flowers/` directory.

## 🎓 Training the Model
To train the model, run:
```bash
python train.py --data_dir flowers --arch vgg16 --epochs 5 --gpu
```

## 🔍 Making Predictions
To classify an image, use:
```bash
python predict.py image.jpg flower_classifier.pth --top_k 5 --category_names cat_to_name.json --gpu
```

## 📊 Visualizing Predictions
A function is included to display an image alongside its **Top-5 Predicted Classes** using Matplotlib.

## 🖥 Technologies Used
- **PyTorch** for deep learning
- **Torchvision** for data transformations
- **Matplotlib** for visualization
- **Jupyter Notebook** for interactive development
- **Command-line scripting** for automation

