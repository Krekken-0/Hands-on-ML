# CNN Fashion-MNIST Classification

A Convolutional Neural Network (CNN) implementation for classifying Fashion-MNIST dataset using PyTorch. This project demonstrates image classification on 10 different fashion item categories.

## Overview

This project implements a deep learning model to classify images from the Fashion-MNIST dataset, which consists of 70,000 grayscale images of 10 different fashion categories. The model uses a convolutional neural network architecture with multiple convolutional layers, pooling operations, and fully connected layers with dropout regularization.

## Dataset

The Fashion-MNIST dataset contains:
- **Training set**: 60,000 images (split into 55,000 for training and 5,000 for validation)
- **Test set**: 10,000 images
- **Classes**: 10 fashion categories
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

The dataset is automatically downloaded when running the notebook (stored in the `dataset/` directory).

## Model Architecture

The CNN model consists of:

1. **Convolutional Layers**:
   - Conv2d(1 → 64 channels) with ReLU activation
   - MaxPool2d (2x2)
   - Conv2d(64 → 128 channels) with ReLU activation
   - Conv2d(128 → 128 channels) with ReLU activation
   - MaxPool2d (2x2)
   - Conv2d(128 → 256 channels) with ReLU activation
   - Conv2d(256 → 256 channels) with ReLU activation
   - MaxPool2d (2x2)

2. **Fully Connected Layers**:
   - Linear(2304 → 128) with ReLU and Dropout(0.5)
   - Linear(128 → 64) with ReLU and Dropout(0.5)
   - Linear(64 → 10) for final classification

All convolutional layers use:
- Kernel size: 3x3
- Padding: "same" (to preserve spatial dimensions)

## Requirements

The project requires the following Python packages:

```
torch
torchvision
torchmetrics
numpy
matplotlib
```

You can install the dependencies using:

```bash
pip install torch torchvision torchmetrics numpy matplotlib
```

## Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook cnn_fashion_mnist.ipynb
   ```

2. **Run the cells sequentially**:
   - The notebook will automatically download the Fashion-MNIST dataset if not already present
   - The model will be trained for 50 epochs
   - Training progress (loss and accuracy) will be displayed for each epoch
   - Final test accuracy will be computed after training

3. **Key Components**:
   - **Model Definition**: Defines the CNN architecture
   - **Training Function**: Implements the training loop with validation
   - **Evaluation Function**: Evaluates model performance on test data
   - **Data Loading**: Loads and preprocesses Fashion-MNIST dataset

## Training Configuration

- **Optimizer**: NAdam (learning rate: 2e-3)
- **Loss Function**: CrossEntropyLoss
- **Metric**: Accuracy (multiclass, 10 classes)
- **Batch Size**: 128
- **Epochs**: 50
- **Device**: Automatically uses CUDA if available, otherwise CPU

## Results

The model achieves high accuracy on the Fashion-MNIST dataset. Training progress shows:
- Decreasing training loss over epochs
- Increasing training and validation accuracy
- Final test accuracy is computed after training completion

## Project Structure

```
CNN-Fashion-Mnist/
├── cnn_fashion_mnist.ipynb    # Main Jupyter notebook
├── dataset/                    # Dataset directory (auto-created)
│   └── FashionMNIST/
│       └── raw/               # Raw dataset files
└── README.md                  # This file
```

## Features

- ✅ Automatic dataset download
- ✅ GPU support (CUDA) with automatic fallback to CPU
- ✅ Comprehensive training loop with validation
- ✅ Visualization of sample images
- ✅ Dropout regularization to prevent overfitting
- ✅ Reproducible results (random seed set)

## Notes

- The dataset is automatically downloaded to the `dataset/` directory on first run
- The model uses dropout (0.5) in fully connected layers for regularization
- Training history (losses and metrics) is tracked and can be used for visualization
- The model architecture is designed to handle 28x28 grayscale images

## License

This project is part of the "Hands-on-ML" learning series and is intended for educational purposes.

