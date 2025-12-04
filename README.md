# Réseau de Neurones DIY (Do It Yourself)

A **from-scratch implementation** of a neural network library in Python using only NumPy. This project implements core deep learning components including feedforward networks, convolutional layers, various activation functions, loss functions, and optimizers—all without relying on high-level frameworks like PyTorch or TensorFlow.

## 🎯 Project Overview

This project demonstrates a deep understanding of neural network fundamentals by implementing:
- **Core Modules**: Linear layers, activation functions (TanH, Sigmoid, Softmax, LogSoftmax)
- **Convolutional Neural Networks**: Conv2D, MaxPool2D, Flatten layers
- **Loss Functions**: MSE, Cross-Entropy, BCE, NLL
- **Optimizers**: SGD with custom backpropagation
- **Advanced Architectures**: Autoencoders, Sequential models
- **Gradient Verification**: Numerical gradient checking for correctness

## ✨ Features

- ✅ **Pure NumPy Implementation**: No deep learning frameworks used for core functionality
- ✅ **Complete Backpropagation**: Custom gradient computation for all layers
- ✅ **CNN Support**: Convolutional and pooling layers for image processing
- ✅ **Multiple Loss Functions**: MSE, Cross-Entropy, BCE, NLL
- ✅ **Flexible Architecture**: Modular design allows easy model construction
- ✅ **Gradient Checking**: Numerical verification ensures mathematical correctness
- ✅ **Tested on Real Data**: MNIST classification and autoencoder implementations
- ✅ **Extensive Testing**: Gradient checks, synthetic data tests, and real-world benchmarks

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Testing](#testing)
- [Results](#results)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/BidoElguindy/Reseau-de-neurones-DIY.git
cd Reseau-de-neurones-DIY
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Reseau-de-neurones-DIY/
├── projet_etu.py          # Main library with all neural network components
├── tests.ipynb            # Comprehensive tests and examples
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git ignore file
```



## 🏗️ Architecture

### Core Components

#### Modules
- **Linear**: Fully connected layer with learnable weights and biases
- **Conv2D**: 2D convolutional layer with customizable kernel size, stride, and padding
- **MaxPool2D**: Max pooling layer for spatial downsampling
- **Flatten**: Reshapes multi-dimensional tensors to vectors

#### Activation Functions
- **TanH**: Hyperbolic tangent activation
- **Sigmoid**: Logistic sigmoid activation
- **Softmax**: Softmax for multi-class classification
- **LogSoftmax**: Log-softmax for numerical stability

#### Loss Functions
- **MSELoss**: Mean Squared Error for regression
- **CrossEntropyLoss**: Cross-entropy for classification with softmax
- **BCE**: Binary Cross-Entropy for binary classification
- **NLLLoss**: Negative Log-Likelihood for use with log-softmax

#### Optimizers & Training
- **Optim**: Optimizer wrapper for training
- **SGD**: Stochastic Gradient Descent with mini-batches

### Design Patterns

The library follows object-oriented principles with a modular design:
- All layers inherit from the base `Module` class
- Forward and backward passes are clearly separated
- Gradient computation is automated through the `Sequentiel` container
- Easy to extend with new layers and loss functions

## 🧪 Testing

The `tests.ipynb` notebook contains comprehensive tests:

1. **Gradient Verification**: Numerical vs. analytical gradients
2. **Simple Regression**: Linear regression on synthetic data
3. **MNIST Classification**: ~97% accuracy on handwritten digits
4. **CNN Tests**: Pattern recognition with custom images
5. **Autoencoder**: MNIST reconstruction
6. **Learning Rate Experiments**: Comparing different learning rates
7. **Multi-setup Tests**: Testing different activation/loss combinations

Run the tests:
```bash
jupyter notebook tests.ipynb
```

## 📊 Results

### MNIST Classification
- **Accuracy**: ~97% on test set
- **Architecture**: 784 → 128 → 10 with TanH activation
- **Training**: 100 epochs, batch size 64, learning rate 0.001

### Gradient Verification
All layers pass numerical gradient checks with relative error < 1e-4

### Autoencoder
Successfully reconstructs MNIST digits with MSE loss < 0.01

### Binary Classification (Moons Dataset)
- **Sigmoid + BCE**: ~85% accuracy
- **Softmax + CrossEntropy**: ~85% accuracy
- **LogSoftmax + NLL**: ~85% accuracy

## 🛠️ Implementation Details

### Backpropagation
Each module implements:
- `forward(X)`: Forward pass computation
- `backward_update_gradient(input, delta)`: Gradient computation for parameters
- `backward_delta(input, delta)`: Gradient propagation to previous layer
- `update_parameters(lr)`: Parameter update with learning rate
- `zero_grad()`: Reset gradients to zero

### Convolutional Layers
- Naive loop-based implementation (no im2col optimization)
- Supports arbitrary kernel sizes, strides, and padding
- Proper gradient computation for both weights and input

### Numerical Stability
- Gradient clipping in activations
- Epsilon values in logarithms to prevent log(0)
- Input normalization in softmax

## 👥 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation


## 🙏 Acknowledgments

This project was developed as part of a machine learning course to understand the fundamentals of neural networks from the ground up.

## 📧 Contact

For questions or feedback:
- GitHub: [@BidoElguindy](https://github.com/BidoElguindy)
- Repository: [Reseau-de-neurones-DIY](https://github.com/BidoElguindy/Reseau-de-neurones-DIY)

---

**Note**: This is an educational project. For production use, consider established frameworks like PyTorch or TensorFlow.