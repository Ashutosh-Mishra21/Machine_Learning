# Neural Network with L2 Regularization

## Overview

This project implements a basic feedforward neural network from scratch using NumPy. The network is designed for binary classification, and it includes L2 regularization to prevent overfitting.

### Key Features:
- Custom forward propagation function with a sigmoid activation.
- Backpropagation algorithm with gradient descent for weight and bias updates.
- L2 regularization to reduce overfitting by penalizing large weights.
- Simple dataset generation for testing purposes.
- Visualization of the loss function over epochs.

## Code Breakdown

### Main Functions

- **forward(X, weights, biases)**: Performs forward propagation through the network layers.
- **sigmoid(z)**: Sigmoid activation function used in the neural network.
- **sigmoid_derivative(z)**: Computes the derivative of the sigmoid function, used during backpropagation.
- **compute_cost(y, y_pred, weights, lambda_reg)**: Computes the cross-entropy loss function with L2 regularization.
- **backward(X, y, weights, biases, activations, Z, lambda_reg)**: Implements backpropagation, calculating the gradients of weights and biases.
- **update_params(weights, biases, dW, db, learning_rate)**: Updates weights and biases using the gradients computed during backpropagation.
- **train(X, y, layer_sizes, epochs, learning_rate, lambda_reg)**: Trains the neural network over a specified number of epochs, returning the final weights and biases.
- **predict(X, weights, biases)**: Uses the trained network to make predictions.

## Installation

1. Clone the repository.
2. Install the necessary dependencies, primarily `numpy` and `matplotlib`.
3. Run the provided notebook or Python script to train the neural network.

```bash
pip install numpy matplotlib
