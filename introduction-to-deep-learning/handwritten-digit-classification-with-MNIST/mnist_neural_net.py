# --- Import Libraries ---
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# --- 1. Set Seed for Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- 2. Configuration Parameters ---
BATCH_SIZE = 64          # Number of images per batch
EPOCHS = 10              # Total number of training cycles
LEARNING_RATE = 0.001    # How quickly the model updates weights
DROPOUT_RATE = 0.3       # Prevents overfitting

# --- 3. Load and Preprocess Data ---
# Transform: convert images to tensors and normalize pixel values to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load MNIST dataset
train_dataset_full = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split training dataset into train and validation (80% / 20%)
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

# Data loaders: feed data in batches
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# --- 4. Define Neural Network Model ---
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                          # Flatten 28x28 image to 784 vector
            nn.Linear(28 * 28, 128),               # First hidden layer
            nn.ReLU(),                             # Activation function
            nn.Dropout(DROPOUT_RATE),              # Dropout to prevent overfitting
            nn.Linear(128, 64),                    # Second hidden layer
            nn.ReLU(),                             # Activation function
            nn.Linear(64, 10)                      # Output layer (10 classes)
        )

    def forward(self, x):
        return self.net(x)

# Instantiate the model and move it to GPU if available
model = MLP()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# --- 5. Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()  # Best for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 6. Training the Model ---
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    running_loss, correct, total = 0.0, 0, 0

    # --- Training Loop ---
    for inputs, labels in train_loader:
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Clear gradients from the previous step
        optimizer.zero_grad()

        # Forward pass: get predictions
        outputs = model(inputs)
        # outputs.shape -> [64, 10] for a batch of 64

        # Calculate loss
        loss = criterion(outputs, labels)
        # Example loss: tensor(0.4271)

        # Backward pass: compute gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)  # Get class with highest score
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # --- Validation Loop ---
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():  # No gradients needed during evaluation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_losses.append(val_loss / total)
    val_accs.append(correct / total)

    # Print results for this epoch
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

# --- 7. Plot Loss and Accuracy Curves ---
import os
os.makedirs('outputs', exist_ok=True)  # Create folder to save plots

# Plot loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('outputs/mnist_loss_plot.png')
plt.show()

# Plot accuracy
plt.figure()
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('outputs/mnist_accuracy_plot.png')
plt.show()

# --- 8. Final Test Accuracy ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Print final test accuracy
print(f"\nFinal Test Accuracy: {correct / total:.4f}")  # Example: Final Test Accuracy: 0.9642

# --- 9. Save Trained Model ---
torch.save(model.state_dict(), "outputs/mnist_model.pt")
