# Heart Disease Prediction using Deep Neural Networks

A Deep Learning classification project built with Keras/TensorFlow to predict the presence of heart disease based on patient clinical parameters.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Summary](#dataset-summary)
- [Model Architecture](#model-architecture)
- [Installation & Requirements](#installation--requirements)
- [Pipeline & Usage](#pipeline--usage)
- [Training & Performance](#training--performance)
- [Directory Structure](#directory-structure)

---

## Project Overview
This project processes a tabular heart disease dataset (`heart.csv`) containing 1,025 patient records with 13 key clinical indicators (e.g., age, chest pain type, resting blood pressure, cholesterol level, maximum heart rate). A multi-layer feedforward Neural Network (MLP) with Dropout regularization and Early Stopping is trained to classify patients into positive or negative target categories for heart disease.

---

## Dataset Summary
- **Total Records:** 1,025
- **Features:** 13 clinical features
- **Target Variable:** `target` (0 = No Heart Disease, 1 = Heart Disease Present)
- **Class Distribution:**
  - Class `1` (Disease): 526 patients (51.32%)
  - Class `0` (No Disease): 499 patients (48.68%)

### Features
| Feature Name | Description | Data Type |
| :--- | :--- | :--- |
| `age` | Age in years | Integer |
| `sex` | Sex (1 = male, 0 = female) | Integer |
| `cp` | Chest pain type (0–3) | Integer |
| `trestbps` | Resting blood pressure (mm Hg) | Integer |
| `chol` | Serum cholesterol (mg/dl) | Integer |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) | Integer |
| `restecg` | Resting electrocardiographic results (0–2) | Integer |
| `thalach` | Maximum heart rate achieved | Integer |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) | Integer |
| `oldpeak` | ST depression induced by exercise relative to rest | Float |
| `slope` | Slope of the peak exercise ST segment (0–2) | Integer |
| `ca` | Number of major vessels (0–4) colored by fluoroscopy | Integer |
| `thal` | Thalassemia status (0–3) | Integer |

---

## Model Architecture
The model is constructed using Keras `Sequential` API:
Input (13 features)
│
├── Dense (8 units, ReLU, He-Normal)
│
├── Dense (4 units, ReLU, He-Normal)
│
├── Dropout (rate = 0.4)
│
├── Dense (2 units, ReLU, He-Normal)
│
└── Dense (1 unit, Sigmoid)


- **Total Parameters:** 161 (all trainable)
- **Loss Function:** `binary_crossentropy`
- **Optimizer:** `Adam`
- **Metrics:** `accuracy`
- **Callbacks:** `EarlyStopping` (monitors `val_loss`, patience = 5, min_delta = 0.001, restores best weights)

---

## Installation & Requirements

### Dependencies
Install the required packages using Python 3.8+:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
Pipeline & Usage
Data Loading: Read heart.csv using pandas.

Train/Test Split: 80% train (820 samples) / 20% test (205 samples) split with random_state=42.

Feature Scaling: StandardScaler fitted on x_train and applied to both train and test splits.

Validation Split during Training: 20% of training data used for validation (656 training / 164 validation).

Model Training

---

## Training & Performance

- **Early Stopping Triggered:** Epoch 89
- **Best Epoch:** Epoch 84 (Restored)
- **Best Validation Accuracy:** **87.20%**
- **Best Validation Loss:** **0.3985**
- **Training Accuracy at Best Epoch:** **90.24%**
- **Training Loss at Best Epoch:** **0.2796**

---

## Directory Structure
.
├── heart.csv            # Input dataset
├── main.ipynb           # Model pipeline notebook
├── README.md            # Project documentation
├── OUTPUT.md            # Execution metrics & evaluation output
└── SUMMARY.html         # Interactive visual HTML summary report