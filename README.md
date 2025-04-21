# CS231n Assignment 1

This repository contains implementations of selected assignments from Stanford's renowned course: [CS231n - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/). These assignments focus on understanding the foundational components of modern deep learning models.

## Contents

| Notebook        | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `knn.ipynb`     | K-Nearest Neighbors classifier: a simple, non-parametric model.            |
| `svm.ipynb`     | Linear SVM classifier: naive and vectorized loss, gradient computation, and training using SGD. |
| `softmax.ipynb` | Softmax classifier: linear classification with vectorized loss & gradients.|
| `learn_features.ipynb` | End-to-end training of a two-layer neural network from scratch and with no external modules, using HOG + color histogram features and cross-validation to find the best model. |

## Key Concepts Covered

- K-nearest neighbors classification
- Linear classification with SVM and Softmax
- Hinge loss and margin-based classification
- Naive vs. vectorized gradient implementation
- Gradient checking and backpropagation
- Feature extraction using HOG and color histograms
- Two-layer neural network implementation using NumPy (no deep learning libraries)
- Mini-batch stochastic gradient descent (SGD)
- Hyperparameter tuning and validation accuracy tracking
- Final test-time evaluation with best saved model

## Technologies

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

## Getting Started

Clone the repository and run the notebooks using Jupyter:

```bash
git clone https://github.com/yourusername/cs231n-assignments.git
cd cs231n-assignments
jupyter notebook
