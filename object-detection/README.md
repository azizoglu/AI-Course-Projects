# 🧠 Object Detection & Real–World Applications with YOLO

Welcome! This repository provides a hands-on introduction to Object Detection using modern deep learning tools, datasets, and frameworks. We cover everything from foundational theory to real-world implementations using YOLO and related tools.

You’ll explore annotation workflows, understand key object detection algorithms, evaluate model performance using common metrics, and build real-world applications such as Rock–Paper–Scissors recognition and balloon detection for defense scenarios.


---

## 📂 Project Structure

```text
.
├── train.py                  # Train your YOLO model for object detection or segmentation using custom datasets
├── predict.py               # Run inference on images, videos, or URLs using a trained YOLO model
├── predict_from_webcam.py   # Perform real-time object detection using your webcam and a trained model
├── requirements.txt         # List of Python dependencies required to run the project
└── README.md                # Project documentation (you're reading it!)
```
---

## 📚 Contents

### 🎯 Object Detection Fundamentals

- **What is Object Detection?** — Understand the task of detecting and localizing objects within images. Explore real-life applications such as autonomous vehicles, surveillance, medical imaging, and industrial inspection.
- **Annotation Formats & Tools** — Learn how to label data for object detection using formats like YOLO, Pascal VOC, and COCO.
Annotation Tools introduced:
- **Annotation Formats & Tools**  
  Learn how to label data for object detection using formats like **YOLO**, **Pascal VOC**, and **COCO**.  
  Annotation Tools introduced:
  - [Roboflow](https://roboflow.com/)
  - [Label Studio](https://labelstud.io/)
  - [CVAT (Computer Vision Annotation Tool)](https://cvat.ai/)

---

### 🔎 Object Detection Algorithms

We explore both classical and deep learning–based detection methods:
- **Classical Methods**  
  - Sliding Window + HOG + SVM  

- **Deep Learning Based**  
  - Region-based Convolutional Neural Network (R-CNN)  
  - Fast R-CNN  
  - Faster R-CNN  
  - SSD (Single Shot MultiBox Detector)  
  - YOLO (You Only Look Once) – [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/#overview)

---

### 📈 Performance Metrics

Understand key evaluation metrics used in object detection:

- **IoU (Intersection over Union)** – Measures overlap between predicted and ground truth boxes  
- **mAP (mean Average Precision)** – Measures overall accuracy across classes and thresholds  
- **Confidence Score** – Indicates model certainty about detections

---

#### ✊ Rock–Paper–Scissors Detection (with Roboflow)
Use a publicly available dataset from **Roboflow** to train a YOLO-based object detection model that identifies hand gestures in images.

You will:
- Import and prepare datasets via Roboflow  
- Train a YOLO model on gesture images  
- Perform real-time detection using webcam input

---

#### 🎈 Balloon Detection System (Inspired by Teknofest)
Simulate a **balloon detection** system for defense/surveillance using aerial images. The dataset is annotated with **Label Studio** and used to train a YOLO model to identify balloons in challenging environments.

You will:
- Annotate data using Label Studio  
- Train and evaluate a YOLO model for balloon detection

---

## 🔗 Recommended Resources
Key references to support your learning on transfer learning, image classification, data augmentation, and Grad-CAM visualization:

### 📘 Core References:

- [YOLO Models (Ultralytics)](https://docs.ultralytics.com/models/yolo11/#overview) 
- [Roboflow](https://roboflow.com/)  
- [Label Studio](https://labelstud.io/)

### 🎥 Video:

- [YOLOv11 Instance Segmentation on Custom Dataset | Step-by-Step Guide (YouTube)](https://www.youtube.com/watch?v=3LN23XJC28U&t=1s&ab_channel=TheCodingBug)


## 🚀 Getting Started

### ✅ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/azizoglu/AI-Course-Projects.git
cd object-detection
pip install -r requirements.txt
