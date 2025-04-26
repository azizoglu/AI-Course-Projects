# 🧠 Transfer Learning, Real–Fake Image Detection & Grad‑CAM

Welcome! In this repository, you'll find a practical introduction to advanced deep learning concepts using **PyTorch**, focused on real-world tasks such as **Transfer Learning**, **Real vs Fake Image Classification**, **Data Augmentation**, and **Model Explainability with Grad-CAM**.  
You will fine-tune powerful pre-trained CNNs, classify AI-generated images, boost model generalization with augmentation, and visualize model attention through heatmaps.


---

## 📂 Project Structure
.
├── real-vs-fake/                    # put your train/val/test folders in here
├── image_classification.py  # training & evaluation pipeline 
├── grad_cam_inference.py    # Grad-CAM heat-map generator
├── predict_single_image.py  # test model with single image
├── requirements.txt         # pip install -r requirements.txt
└── README.md                # ← you are here

---

## 📚 Contents

### 🔍 Theoretical Overview

- **Transfer Learning** — Accelerate model training and achieve high accuracy by adapting pre-trained CNNs to new tasks.
- **Real vs Fake Image Classification** — Build robust models capable of distinguishing between authentic and AI-generated images.
- **Data Augmentation** — Enhance model generalization by dynamically transforming training data through diverse techniques.
- **Grad-CAM Explainability** — Visualize and interpret the critical regions your network focuses on during predictions.

---

#### 📗 Example: Real vs Fake Image Classification
**File**: `image_classification.py`  
Fine-tune a pre-trained Convolutional Neural Network (CNN) to classify real versus AI-generated (fake) images.

What you'll learn:
- Apply **Transfer Learning** using models like VGG19 or ResNet
- Perform **Data Augmentation** to improve model generalization
- Train, validate, and monitor model performance with live metrics (accuracy, loss)
- Save and load trained models using `torch.save` and `torch.load`

---

#### 📕 Single Image Prediction
**File**: `predict_single_image.py`  
Quickly load a trained model and make a prediction on a single image (real or fake).

What you'll learn:
- Preprocess and transform a single input image to match the model requirements
- Load a saved model checkpoint using `torch.load`
- Perform forward inference and obtain the predicted class and confidence score
- Visualize or print the prediction result for quick testing and validation

---

#### 📘 Example: Grad-CAM Visualization
**File**: `grad_cam_inference.py`  
Generate Grad-CAM heatmaps to visualize which regions of an image influence the model’s predictions the most.

What you'll learn:
- Load a trained model and register hooks to extract intermediate feature maps
- Perform a forward and backward pass to compute Grad-CAM activations
- Generate and overlay class-specific heatmaps on the original image
- Interpret model behavior by identifying focus areas during prediction

---

### 🏠 Practice Assignment

#### 🎯 Task: Plant Disease Classification
**File**: `plant_disease_assignment.py`  
Apply your deep learning skills on a new challenge — classifying plant leaf images into **Healthy**, **Powdery Mildew**, or **Rust** categories.

You will:
- Fine-tune a pre-trained CNN model for multi-class classification
- Perform **Data Augmentation** (e.g., random crop, flip, color jitter) to improve robustness
- Use activation functions like ReLU and Softmax for multi-class outputs
- Experiment with different optimizers, learning rates, and model architectures
- Save your best-performing model for future inference

---

## 🔗 Recommended Resources
Key references to support your learning on transfer learning, image classification, data augmentation, and Grad-CAM visualization:

### 📘 Core References:

- [TorchVision Pre-trained Models](https://pytorch.org/vision/main/models.html) — Overview of popular pre-trained CNN architectures.
- [ImageNet Dataset](https://www.image-net.org/) — Benchmark dataset for image recognition and transfer learning.
- [Transfer Learning Tutorial — PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — Step-by-step guide to applying transfer learning with PyTorch.
- Selvaraju, R. R., et al. (2017). [Grad-CAM: Visual explanations from deep networks via gradient-based localization](https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html). *ICCV 2017* — Original Grad-CAM paper.
- Shorten, C., & Khoshgoftaar, T. M. (2019). [A survey on image data augmentation for deep learning](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0). *Journal of Big Data* — Comprehensive review on data augmentation techniques.

### 🎥 Video:

- [Convolutional Neural Networks (DeepLearning.AI - Andrew Ng)](https://www.youtube.com/watch?v=yofjFQddwHE&ab_channel=DeepLearningAI) — Practical introduction to CNNs and feature extraction.


## 🚀 Getting Started

### ✅ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/azizoglu/AI-Course-Projects.git
cd transfer-learning-and-image-classification
pip install -r requirements.txt