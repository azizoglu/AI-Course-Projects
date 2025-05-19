import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100             # Size of the input noise vector
batch_size = 128             # Training batch size
epochs = 100                 # Number of training epochs
img_size = 28                # Size of MNIST images (28x28)
img_shape = (1, 28, 28)      # Image shape: 1 channel, 28x28
lr = 0.0002                  # Learning rate
save_interval = 10           # Save image samples every N epochs

# Load and normalize the MNIST dataset (-1 to 1 range)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

dataset = datasets.MNIST(root="mnist_data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator model: transforms random noise into a fake image
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), *img_shape)  # Reshape to (N, 1, 28, 28)
        return img

# Discriminator model: classifies images as real or fake
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output: probability of real
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)  # Flatten the image
        validity = self.model(flat)
        return validity

# Initialize models and move to the selected device
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function: Binary Cross Entropy
adversarial_loss = nn.BCELoss()

# Optimizers for both networks
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Function to save generated image samples
def save_images(epoch):
    z = torch.randn(25, latent_dim).to(device)       # Generate 25 random noise vectors
    gen_imgs = generator(z).detach().cpu()           # Generate images and detach from graph
    gen_imgs = (gen_imgs + 1) / 2                    # Rescale from [-1,1] to [0,1]

    fig, axs = plt.subplots(5, 5, figsize=(5, 5))     # 5x5 image grid
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[cnt][0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.suptitle(f"Epoch {epoch}")
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/mnist_{epoch}.png")
    plt.close()

# Start training the GAN
for epoch in range(1, epochs + 1):
    for i, (imgs, _) in enumerate(dataloader):

        # Real images and labels (1 = real)
        real_imgs = imgs.to(device)
        valid = torch.ones(imgs.size(0), 1, device=device)
        fake = torch.zeros(imgs.size(0), 1, device=device)

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim, device=device)  # Random noise
        gen_imgs = generator(z)                                   # Generate fake images
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)  # Try to fool the discriminator
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)  # Loss on real images
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)  # Loss on fake images
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    # Print training progress
    print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Save image samples every few epochs
    if epoch % save_interval == 0:
        save_images(epoch)
