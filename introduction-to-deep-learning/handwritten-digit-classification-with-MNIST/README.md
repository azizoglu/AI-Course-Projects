# 🧠 MNIST Digit Classification with PyTorch

This project demonstrates how to train a simple Multi-Layer Perceptron (MLP) to classify handwritten digits from the MNIST dataset using PyTorch.

![MNIST Dataset](../images/mnist-dataset.png)

---

## 📦 1. Import Libraries

We start by importing essential libraries:
- `torch`, `torchvision` for building and training neural networks.
- `matplotlib` for visualization.
- `numpy` and `random` for numerical operations and reproducibility.

---

## 🔁 2. Set Seed for Reproducibility

We set the random seed for Python, NumPy, and PyTorch to ensure consistent results across runs.

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

---

## ⚙️ 3. Configuration Parameters

Key training parameters are defined:
- `BATCH_SIZE`: Number of images per batch
- `EPOCHS`: Total training cycles
- `LEARNING_RATE`: Step size for optimizer
- `DROPOUT_RATE`: Dropout rate to prevent overfitting

---

## 🧹 4. Load and Preprocess Data

- Normalize images to the range `[-1, 1]`.
- Load MNIST dataset and split training data into 80% training and 20% validation.
- Wrap datasets in `DataLoader` for efficient batching.

---

## 🧠 5. Define the Neural Network

A simple **MLP (Multi-Layer Perceptron)** architecture:
- Input: Flattened 28x28 image → 784 units
- Hidden Layer 1: 128 units + ReLU + Dropout
- Hidden Layer 2: 64 units + ReLU
- Output: 10 units (digits 0–9)

```python
class MLP(nn.Module):
    ...
```

The model is moved to GPU if available.

---

## ⚙️ 6. Define Loss and Optimizer

- **Loss**: Cross-Entropy Loss (good for multi-class classification)
- **Optimizer**: Adam (adaptive gradient descent)

---

## 🏋️‍♂️ 7. Train the Model

For each epoch:
- **Training loop**: Forward pass, loss calculation, backward pass, optimizer step.
- **Validation loop**: Evaluate model without updating weights.
- Track and store loss and accuracy for both training and validation sets.

Progress is printed after each epoch.

---

## 📈 8. Plot Loss and Accuracy

Two plots are generated and saved:
- **Loss vs Epoch**
- **Accuracy vs Epoch**

These help visualize the model’s learning progress.

---

## 🧪 9. Evaluate on Test Set

- After training, the model is tested on the unseen **test dataset**.
- Final test accuracy is calculated and printed.

---

## 💾 10. Save the Model

The trained model's parameters are saved to `outputs/mnist_model.pt` for future use or inference.

---

## ✅ Output Example

```
Epoch 10/10 | Train Loss: 0.1453, Train Acc: 0.9568 | Val Loss: 0.1394, Val Acc: 0.9592

Final Test Accuracy: 0.9642
```

---
