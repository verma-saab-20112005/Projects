# F1 Pit Stop Strategy Prediction using Deep Learning

This repository contains an end-to-end Machine Learning and Neural Network workflow designed to predict Formula 1 pit stop decisions (`PitStop` binary target) based on race dynamics, telemetry deltas, driver indicators, and tyre degradation profiles.

## Table of Contents
- [Overview](#overview)
- [Dataset Architecture](#dataset-architecture)
- [Model Architecture & Training](#model-architecture--training)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)

---

## Overview

Formula 1 strategy relies heavily on timing pit stops according to tyre wear, lap time degradation, track position, and race progress. This project processes lap-by-lap telemetry data from F1 Grand Prix races (2023–2024) and trains a Keras Artificial Neural Network (ANN) with regularized dense layers to accurately predict pit stop timing.

---

## Dataset Architecture

The raw dataset (`f1_strategy_dataset_v2.csv`) contains **52,471 records** across **16 initial columns**:

### Feature Set
- **Categorical Columns**: `Driver` (25 drivers), `Compound` (HARD, MEDIUM, SOFT, INTERMEDIATE, WET), `Race` (25 Grand Prix events).
- **Numerical Features**: `LapNumber`, `Stint`, `TyreLife`, `Position`, `LapTime (s)`, `Year`, `LapTime_Delta`, `Cumulative_Degradation`, `PitNextLap`, `RaceProgress`, `Normalized_TyreLife`, `Position_Change`.
- **Target Variable**: `PitStop` (0: No pit stop = 42,627 samples, 1: Pit stop = 9,844 samples).

### Feature Preprocessing
1. **One-Hot Encoding**: Categorical variables (`Driver`, `Compound`, `Race`) are transformed via `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, expanding the feature space to **67 total features**.
2. **Standardization**: Features are scaled using Scikit-Learn’s `StandardScaler`.
3. **Train-Test Split**: 80/20 train-test split (`x_train`: 41,976 samples).

---

## Model Architecture & Training

The binary classification model is implemented using Keras `Sequential`:

```python
Model: "Sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Input                        (None, 67)                0         
Dense (64, ReLU, L2=0.002)   (None, 64)                4,352     
Dense (32, ReLU, L2=0.002)   (None, 32)                2,080     
Dense (16, ReLU, L2=0.001)   (None, 16)                528       
Dropout (0.5)                (None, 16)                0         
Dense (4, ReLU, L2=0.001)    (None, 4)                 68        
Dense (2, ReLU, L2=0.001)    (None, 2)                 10        
Dense (1, Sigmoid, L2=0.001) (None, 1)                 3         
=================================================================
===============================================================
Optimizer: Adam

Loss Function: Binary Crossentropy

Callbacks: EarlyStopping monitoring val_loss (patience=5, min_delta=0.001)

Batch Size: 32