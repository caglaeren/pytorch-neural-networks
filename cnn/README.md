# Convolutional Neural Network (CNN)

This folder contains an implementation of a Convolutional Neural Network (CNN)
using PyTorch. CNNs are particularly effective for image data due to their
ability to capture spatial features such as edges, textures, and shapes.

## Model Overview
- Architecture: Convolutional Neural Network
- Framework: PyTorch
- Task: Image classification
- Dataset: CIFAR-10 (or MNIST if grayscale images are used)

## Key Concepts
- Convolutional layers
- ReLU activation
- Max pooling
- Fully connected layers
- Softmax classification

## Dataset
The model is trained and evaluated on an image classification dataset:
- **CIFAR-10**: 60,000 color images (32×32) across 10 classes  
  *(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)*

## Training Details
- Loss function: CrossEntropyLoss
- Optimizer: SGD / Adam
- Evaluation metric: Accuracy
- Epochs: configurable

## How to Run
```bash
python cnn.py
```

## Learning Objectives
- Understand how convolutional layers extract spatial features
- Learn how CNNs differ from fully connected networks (ANNs)
- Gain practical experience with image classification using PyTorch
