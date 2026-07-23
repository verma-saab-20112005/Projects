# Student Placement Status Prediction Model

## Project Overview
This project applies machine learning to predict whether a student will be **Placed** or **Not Placed** based on their academic, technical, and interpersonal profiles. It utilizes an **XGBClassifier** model optimized via 5-fold grid search cross-validation and encapsulated using a Scikit-Learn pipeline.

---

## Technical Pipeline & Implementation

1. **Preprocessing Pipeline** (`Pipeline.pkl`):
   - **Numerical Features** (`CGPA`, `Aptitude_Test_Score`, `Projects`, etc.): Median imputation followed by standard scaling (`StandardScaler`).
   - **Categorical Features** (`Gender`, `Degree`, `Branch`): Constant imputation with `"missing"` followed by one-hot encoding (`OneHotEncoder(handle_unknown="ignore")`).
2. **Target Labeling**:
   - Encoded using `LabelEncoder()` to map binary placement categories to numerical targets.
3. **Model Selection & Tuning**:
   - Classifier: `XGBClassifier(objective="multi:softmax", random_state=42)`
   - Hyperparameter Optimization: `GridSearchCV` across parameters `n_estimators`, `learning_rate`, `max_depth`, `subsample`, and `colsample_bytree`.
   - Artifacts Saved: `Training_model.pkl` and `Pipeline.pkl`.

---

## File Architecture
* `train.py` / `train_2.ipynb`: Scripts/notebooks for data preprocessing, pipeline assembly, model training, and model serialization.
* `test.py` / `test.ipynb`: Scripts/notebooks for pipeline deserialization, model loading, generating predictions on `test.xlsx`, and computing test accuracy.
* `Pipeline.pkl`: Serialized `ColumnTransformer` preprocessing object.
* `Training_model.pkl`: Serialized best-performing XGBoost model.
* `train.xlsx`: Training dataset containing 45,000 student records.
* `test.xlsx`: Validation test dataset containing 5,000 student records.