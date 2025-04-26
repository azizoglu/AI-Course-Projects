# 🧠 Introduction to Deep Learning with PyTorch

Welcome! In this repository, you'll find a beginner-friendly introduction to deep learning using **PyTorch**, including both theoretical background and hands-on coding examples. You'll build and train neural networks on image datasets like **MNIST**, **Fashion MNIST**, and **CIFAR-10**.

---

## 📚 Contents

### 🔍 Theoretical Overview

Learn the fundamentals of deep learning:
- What makes deep learning powerful?
- Neural network components: neurons, layers, activations
- Training process: forward pass, backpropagation, optimization
- Preventing overfitting: dropout, regularization, early stopping
- Popular frameworks: PyTorch vs TensorFlow

---

### 🧪 Hands-On Examples

#### 📘 Example A: Handwritten Digit Classification (MNIST)
**File**: `mnist_neural_net.py`  
Build a basic MLP (Multilayer Perceptron) to classify handwritten digits (0–9).

What you'll learn:
- How to load and preprocess MNIST using `torchvision.datasets`
- Create a neural network with `torch.nn.Sequential`
- Train using cross-entropy loss and the Adam optimizer
- Add dropout and test different learning rates

#### 📗 Example B: Image Classification with CIFAR-10
**File**: `cifar10_classifier.py`  
Use a Convolutional Neural Network (CNN) to classify CIFAR-10 images (e.g., cats, airplanes, cars).

What you'll learn:
- Load and transform RGB image data with data augmentation
- Build a CNN using `torch.nn` layers like `Conv2d`, `MaxPool2d`, `BatchNorm2d`
- Visualize model performance over epochs
- Save and load models with `torch.save` and `torch.load`

---

### 🏠 Practice Assignment

#### 🎯 Task: Fashion MNIST Classification
**File**: `fashion_mnist_assignment.py`  
Practice your skills on a new dataset — Fashion MNIST (e.g., shirts, shoes, bags).

You will:
- Build and train an MLP for fashion item classification
- Use ReLU + Softmax (or LogSoftmax) activations
- Experiment with different optimizers and hyperparameters
- Add dropout and increase model depth
- Save your trained model for later use

---

## 🔗 Recommended Resources
Below are some helpful resources and references—both written and video—to reinforce your understanding of deep learning and PyTorch:

### Video Explanations:

- [Neural Network In 5 Minutes | What Is A Neural Network? | How Neural Networks Work — Simplilearn](https://youtu.be/bfmFfD2RIcg?si=YljxQ3pw9zmqVBzJ)

- [What Do Neural Networks Really Learn? Exploring the Brain of an AI Model — Rational Animations](https://youtu.be/jGCvY4gNnA8?si=5qSZARRaiqv3bCr8)

- [But what is a neural network? | Deep learning chapter 1 — 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&t=110s)

### PyTorch & Deep Learning:

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)

- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

- [Andrew Ng – Deep Learning Specialization](https://www.deeplearning.ai/)

---

## 🚀 Getting Started

### ✅ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/azizoglu/AI-Course-Projects.git
cd introduction-to-deep-learning
pip install -r requirements.txt
