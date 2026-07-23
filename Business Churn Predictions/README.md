# Customer Churn Prediction Engine

An end-to-end Machine Learning pipeline designed to predict customer churn using an XGBoost Classifier paired with automated preprocessing and hyperparameter tuning (`GridSearchCV`).

---

## Technical Overview

This repository builds a high-accuracy classification pipeline to identify at-risk customers from raw tabular data.

### Pipeline Architecture
[ Raw CSV Data ]
                                     │
                     Stratified Train/Test Split (80/20)
                                     │
                ┌────────────────────┴────────────────────┐
                ▼                                         ▼
       Categorical Features                      Numerical Features
                │                                         │
  SimpleImputer(constant="missing")             SimpleImputer(strategy="median")
                │                                         │
        OneHotEncoder                               StandardScaler
                │                                         │
                └────────────────────┬────────────────────┘
                                     ▼
                           [ ColumnTransformer ]
                                     │
                          GridSearchCV (5-Fold CV)
                                     │
                            [ XGBoost Model ]
                                     │
                          Predictions & Metrics

### Preprocessing & Feature Pipeline
* **Categorical Handling:** Imputes missing strings as `"missing"` and applies one-hot encoding (`handle_unknown="ignore"`).
* **Numerical Handling:** Imputes missing numerical values using median strategies and normalizes values with standard scaling (`StandardScaler`).
* **Stratified Data Splitting:** Ensures identical class distribution (`churn`) across training and test subsets using `StratifiedShuffleSplit`.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `churn_model.py` | Full execution script containing preprocessing pipeline, grid search, and accuracy scoring. |
| `exploration.ipynb` | Exploratory Data Analysis (EDA) notebook displaying raw feature schemas and categorical distributions. |
| `customer_churn_business_dataset.csv` | Dataset containing customer demographic, usage, support, and billing features. |

---

## Model Configuration & Hyperparameters

The model uses `GridSearchCV` across a 5-fold cross-validation scheme to optimize the following search grid for `XGBClassifier`:

```python
param_grid = {
    'n_estimators': [100, 700],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
Quick Start & Setup
Prerequisites
Python 3.8+

pandas

numpy

scikit-learn

xgboost