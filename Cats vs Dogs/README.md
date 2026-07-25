# Cats vs Dogs Image Classification using Convolutional Neural Networks (CNN)

This project builds, trains, and evaluates a Deep Convolutional Neural Network (CNN) using TensorFlow/Keras to perform binary image classification (Cats vs. Dogs).

> **Note**: The raw image dataset is not included in this repository due to file size constraints.

---

## 📌 Project Overview
- **Objective**: Classify input images of size $256 \times 256 \times 3$ as either a Cat or a Dog.
- **Dataset Split**:
  - **Training Set**: 791 images across 2 classes
  - **Testing Set**: 209 images across 2 classes
- **Framework**: TensorFlow 2.x / Keras
- **Input Resolution**: $256 \times 256 \times 3$ (RGB)
- **Normalization**: Rescaled pixel values from $[0, 255]$ to $[0, 1]$

---

## 🏗️ Model Architecture

The model is built using a Keras `Sequential` architecture consisting of 3 Convolutional blocks followed by a fully connected (Dense) classification head.

1. **Conv Block 1**: `Conv2D` (32 filters, 3x3) $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling2D` (2x2)
2. **Conv Block 2**: `Conv2D` (64 filters, 3x3) $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling2D` (2x2)
3. **Conv Block 3**: `Conv2D` (128 filters, 3x3) $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling2D` (2x2)
4. **Flatten Layer**: Flattens the $30 \times 30 \times 128$ feature map to 115,200 units.
5. **Fully Connected Layers**: Dense layers with sizes [128, 64, 32, 16, 8, 4] using ReLU activations, along with `Dropout` layers (0.3 and 0.2 rate) to prevent overfitting.
6. **Output Layer**: `Dense` (1 unit, Sigmoid activation)

### Model Parameters Summary
- **Total Parameters**: 14,850,913 (56.65 MB)
- **Trainable Parameters**: 14,850,465 (56.65 MB)
- **Non-trainable Parameters**: 448 (1.75 KB)

---

## ⚙️ Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy (`binary_crossentropy`)
- **Metrics**: Accuracy
- **Early Stopping**: Monitored `val_loss` with `patience=5` and `min_delta=0.0001`
- **Batch Size**: 32
- **Epochs Run**: Stopped at Epoch 18/200 via Early Stopping

---

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>

Cats vs Dogs/
├── Train/
│   ├── cats/
│   └── dogs/
└── Test/
    ├── cats/
    └── dogs/