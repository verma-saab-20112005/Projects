# Drug Classification using Deep Neural Network (Keras / TensorFlow)

This repository contains a deep learning pipeline designed to classify drug prescriptions based on patient demographic and clinical attributes (Age, Sex, Blood Pressure, Cholesterol, and Sodium-to-Potassium Ratio).

---

## 📌 Project Overview

The objective is to predict one of five target drug categories (`DrugY`, `drugX`, `drugA`, `drugC`, `drugB`) using a feedforward Artificial Neural Network (ANN) built with Keras/TensorFlow and feature preprocessing powered by Scikit-Learn and Pandas.

---

## 📊 Dataset Profile

- **Source File**: `drug200.csv`
- **Total Instances**: 200 rows
- **Total Features**: 5 input features + 1 target column (`Drug`)
- **Missing Values**: 0 null values across all columns

### Features Breakdown
1. `Age` *(Numerical / int64)*: Age of the patient.
2. `Sex` *(Categorical / object)*: Gender (`F`, `M`).
3. `BP` *(Categorical / object)*: Blood pressure level (`HIGH`, `LOW`, `NORMAL`).
4. `Cholesterol` *(Categorical / object)*: Cholesterol level (`HIGH`, `NORMAL`).
5. `Na_to_K` *(Numerical / float64)*: Sodium-to-potassium ratio in blood.
6. `Drug` *(Target / object)*: Recommended drug type (`DrugY`, `drugX`, `drugA`, `drugC`, `drugB`).

### Target Class Distribution
| Target Class | Count | Percentage |
| :--- | :--- | :--- |
| **DrugY** | 91 | 45.5% |
| **drugX** | 54 | 27.0% |
| **drugA** | 23 | 11.5% |
| **drugC** | 16 | 8.0% |
| **drugB** | 16 | 8.0% |

---

## ⚙️ Data Preprocessing Pipeline

1. **Categorical Encoding**:
   - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` was used to expand `Sex` (2 features), `BP` (3 features), and `Cholesterol` (2 features) into binary dummy variables.
   - Total transformed input features: **9 columns** (`Age`, `Na_to_K`, `Sex_F`, `Sex_M`, `BP_HIGH`, `BP_LOW`, `BP_NORMAL`, `Cholesterol_HIGH`, `Cholesterol_NORMAL`).

2. **Train-Test Split**:
   - Split ratio: **80% Train** (160 samples), **20% Test** (40 samples).
   - `random_state = 42`.

3. **Feature Scaling**:
   - Standard scaling (`StandardScaler`) applied to numerical and one-hot encoded input features.

4. **Target Encoding**:
   - Target labels one-hot encoded into 5 binary vector representations for multi-class cross-entropy loss calculation.

---

## 🧠 Neural Network Architecture
Input Vector (9 features)
│
▼
Dense Layer (7 units, ReLU)
│
▼
Dropout Layer (Rate = 0.3)
│
▼
Dense Layer (5 units, Softmax Output)


### Hyperparameters & Configuration
- **Input Shape**: `(9,)`
- **Hidden Layer**: 7 Neurons with **ReLU** activation.
- **Regularization**: Dropout rate of `0.3`.
- **Output Layer**: 5 Neurons with **Softmax** activation.
- **Optimizer**: `Adam`
- **Loss Function**: `categorical_crossentropy`
- **Metrics**: `accuracy`
- **Early Stopping**: Monitored `val_loss`, `min_delta=0.001`, `patience=5`, stopped training at epoch **123**.

---

## 📈 Model Performance Summary

- **Initial Epoch (Epoch 1)**: Loss: `1.7331` | Accuracy: `28.91%` | Val Loss: `1.7188` | Val Accuracy: `25.00%`
- **Peak Validation Accuracy**: **100.00%** achieved at Epoch **107** (Val Loss: `0.1243`).
- **Lowest Validation Loss**: **0.1135** achieved at Epoch **118** (Val Accuracy: `96.88%`).
- **Early Stopping Trigger**: Epoch **123** (Val Loss: `0.1141`, Val Accuracy: `96.88%`).

---

## 🚀 How to Run

1. Clone or download the repository.
2. Place `drug200.csv` in the working directory.
3. Install required Python packages:
   ```bash
   pip install pandas numpy matplotlib tensorflow scikit-learn