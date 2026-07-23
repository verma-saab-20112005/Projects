# Sleep Health and Lifestyle Analysis & Neural Network Classification

This project processes the **Sleep Health and Lifestyle Dataset** to predict sleep disorders (e.g., *Insomnia*, *Sleep Apnea*, or *None*) based on lifestyle, demographic, and physiological features using a Deep Neural Network built with TensorFlow/Keras.

## Dataset Overview
- **Source File**: `Sleep_health_and_lifestyle_dataset.csv`
- **Total Records**: 374 entries
- **Target Variable**: `Sleep Disorder` (`None`, `Sleep Apnea`, `Insomnia`)

## Features & Preprocessing Pipeline
1. **Target Imputation**: Missing values in `Sleep Disorder` were filled with `"None"`.
2. **Feature Engineering**:
   - `Blood Pressure` (string format like `120/80`) was split into two numerical features: `High BPM` (Systolic) and `Low BPM` (Diastolic).
   - Removed uninformative identifiers (`Person ID`).
3. **One-Hot Encoding**: Applied to categorical variables:
   - `Gender`
   - `Occupation`
   - `BMI Category`
   - Target variable `Sleep Disorder` for multi-class classification.
4. **Train-Test Split**: 80/20 train-test ratio (`test_size=0.2`, `random_state=42`).
5. **Feature Scaling**: Applied `StandardScaler` to transform input features (26 total preprocessed features).

## Neural Network Model Architecture
- **Framework**: TensorFlow / Keras (`Sequential` model)
- **Input Dimension**: 26 preprocessed features
- **Layers**:
  - `Dense(16, activation='relu', kernel_regularizer=L2(0.002))` + `Dropout(0.3)`
  - `Dense(8, activation='relu', kernel_regularizer=L2(0.002))` + `Dropout(0.3)`
  - `Dense(4, activation='relu', kernel_regularizer=L2(0.002))` + `Dropout(0.2)`
  - `Dense(3, activation='softmax', kernel_regularizer=L2(0.002))` (Output Layer)
- **Regularization**: L2 Weight Regularization and Dropout layers to prevent overfitting.
- **Callback**: `EarlyStopping` monitoring `val_loss`.

## Requirements
- `python >= 3.8`
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow` / `keras`