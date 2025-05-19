# 🧠 Generative Adversarial Networks (GANs)

Welcome! This repository provides a foundational and practical introduction to **Generative Adversarial Networks (GANs)** — a powerful class of generative models that learn to create new data by pitting two neural networks against each other in a game-theoretic framework.

---

## 📂 Project Structure

```text
.
├── gan_mnist.py            # GAN implementation for MNIST digit generation
├── requirements.txt        # List of Python dependencies required to run the project
└── README.md              # Project documentation (you're reading it!)
```

---

## 📚 Contents

### 🧩 Core Concepts Covered

Throughout this module, you will explore key GAN principles, including:

- **Generator** and **Discriminator** Networks
- **Adversarial Training**
- **Loss Functions** (Generator and Discriminator Losses)
- **Training Stability**
- **Mode Collapse**
- **Fine-Tuning** and **RAG (Retrieval-Augmented Generation)**

These elements form the basis of how GANs learn to generate realistic data through adversarial training.

---

### 🛠️ GAN Approaches

#### 🔹 Basic GAN Architecture
- Understand the fundamental GAN architecture with Generator and Discriminator networks
- Learn about the adversarial training process
- Implement MNIST digit generation using GANs

#### 🔸 Advanced GAN Variants
- **CycleGAN**: Unpaired image-to-image translation
  - Learn how to transform images between domains without paired examples
  - Understand cycle consistency loss
  - Reference: Zhu et al. (2017) - "Unpaired image-to-image translation using cycle-consistent adversarial networks"

- **StyleGAN**: High-quality image generation
  - Explore style-based generator architecture
  - Understand progressive growing and style mixing
  - Reference: Karras et al. (2019) - "A style-based generator architecture for generative adversarial networks"

---

### 🎨 Interactive Learning

Experience GANs in action through interactive tools:

- [GAN Lab](https://poloclub.github.io/ganlab/) - Interactive visualization of GAN training
  - Visualize the training process in real-time
  - Understand how generator and discriminator networks interact
  - Experiment with different architectures and hyperparameters

---

### 🤖 Fine-Tuning and RAG

Learn about advanced techniques in generative AI:

- **Fine-Tuning**: Adapting pre-trained models to specific tasks
- **RAG (Retrieval-Augmented Generation)**: Enhancing generation with external knowledge
- Practical implementation using AnythingLLM

---

## 🔗 Recommended Resources

- [Microsoft Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners)
  A comprehensive guide to getting started with generative AI, including GANs and other generative models.

- [GAN Lab Documentation](https://poloclub.github.io/ganlab/)
  Interactive visualization tool for understanding GAN training dynamics.

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
  Original paper on CycleGAN by Zhu et al.

- [StyleGAN Paper](https://arxiv.org/abs/1812.04948)
  Original paper on StyleGAN by Karras et al.

---

## 🚀 Getting Started

### ✅ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/azizoglu/AI-Course-Projects.git
cd generative-adversarial-networks
pip install -r requirements.txt
