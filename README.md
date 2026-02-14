# PyTorch Neural Networks

This repository contains implementations of various neural network
architectures using PyTorch. Each model is organized in its own folder
with a dedicated dataset, architecture, and documentation.

The project is designed to progressively cover different neural network
types, starting from basic models and moving toward more advanced
architectures.

## Implemented Models

- **ANN (Artificial Neural Network)**
  - Fully connected feedforward neural network
  - MNIST digit classification
  - Folder: `ann/`

- **CNN (Convolutional Neural Network)**
  - Designed for image and spatial data processing
  - Uses convolutional layers for feature extraction
  - Folder: `cnn/`


> Additional neural network models will be added over time.

## Repository Structure

```text
pytorch-neural-network/
├── README.md
├── ann/
│   ├── ann.py
│   └── README.md
├── cnn/
│   ├── cnn.py
│   └── README.md
├── rnn/              # (planned)
├── lstm/             # (planned)
├── gru/              # (planned)
├── transformer/      # (planned)
```

## Technologies Used
- Python
- PyTorch
- torchvision


## How to Use
Each model directory contains its own `README.md` with:
- Dataset information
- Model architecture description
- Training and evaluation details
- Instructions for running the code


## Learning Objectives
- Understand different neural network architectures
- Gain hands-on experience with PyTorch
- Compare models across different problem types
- Build a scalable and well-structured deep learning codebase

## Status
🚧 This repository is under active development and will continue to grow
with additional neural network models and experiments.
