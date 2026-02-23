# 📈 Time Series Forecasting with RNN (Sine Wave)

This project demonstrates how to use a Recurrent Neural Network (RNN) to perform time series prediction on a synthetic sine wave dataset.
The main goal is to understand how RNNs process sequential data and learn temporal dependencies using PyTorch.

This project is designed for educational purposes and serves as a foundational example for sequence modeling.

## 🚀 Project Objective
- Understand the structure of time series data
- Learn the sequence–target prediction approach
- Build a simple RNN model using PyTorch
- Train the model and visualize prediction results

## 🧠 Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib


## Dataset 
- The dataset is synthetically generated.
- X: Linearly spaced values between 0 and 100
- y: sin(X) function
- Sequence Length: 50 time steps

  Input (sequence):  [x1, x2, x3, ..., x50]
  Target (label):    x51


## ⚙️ Model Architecture
- RNN Layer: Learns temporal dependencies in sequential data
- Fully Connected Layer: Maps RNN output to a single prediction
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam

## How to Run
```bash
python rnn.py
```


