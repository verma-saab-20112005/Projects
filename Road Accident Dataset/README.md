# Indian Roads Accident Risk Prediction Model

This project builds a Deep Learning Regression Model using TensorFlow/Keras to predict road accident **Risk Scores** across major Indian cities[cite: 1]. The model processes spatial, temporal, environmental, and road-condition attributes to quantify risk factors and assist in road safety analytics[cite: 1].

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Characteristics](#dataset-characteristics)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Training & Optimization](#training--optimization)
- [Installation & Usage](#installation--usage)

---

## 📌 Project Overview
- **Task Type**: Regression (Continuous target: `risk_score`)[cite: 1]
- **Framework**: TensorFlow / Keras, Scikit-Learn, Pandas, NumPy[cite: 1]
- **Dataset Size**: 20,000 records, originally 24 features[cite: 1]
- **Final Input Features**: 41 scaled numerical features[cite: 1]

---

## 📌 Dataset Characteristics
The dataset (`indian_roads_dataset.csv`) contains road incident details across key metropolitan cities in India[cite: 1]:
* **Geographical**: Latitude, Longitude, City (Pune, Mumbai, Chandigarh, Chennai, Bangalore, Delhi, Hyderabad, Kolkata)[cite: 1]
* **Temporal**: Date, Hour, Weekend flag, Peak hour flag[cite: 1]
* **Environmental**: Weather conditions (Clear, Fog, Rain), Visibility (Low, Medium, High), Temperature[cite: 1]
* **Traffic & Road**: Road Type (Highway, Rural, Urban), Lanes, Traffic Signals, Traffic Density (Low, Medium, High)[cite: 1]
* **Incident Details**: Primary Cause (Distraction, Overspeeding, Weather, Drunk Driving, Poor Road), Accident Severity (Minor, Major, Fatal), Vehicles Involved, Casualties[cite: 1]
* **Target**: `risk_score` (Continuous score ranging from `0.10` to `1.00`)[cite: 1]

---

## 📌 Data Preprocessing Pipeline
1. **Handling Missing Values**:
   - The `festival` column contained 19,885 missing values out of 20,000 (~99.4%) and was dropped[cite: 1].
2. **Feature Removal**:
   - Dropped low-variance or redundant attributes: `state`, `time`, `day_of_week`, and identifier `accident_id`[cite: 1].
3. **Date Parsing**:
   - Converted string `date` column into `year`, `month`, and `day` numerical features[cite: 1].
4. **Categorical Feature Encoding**:
   - Applied **One-Hot Encoding** (`OneHotEncoder(sparse_output=False)`) on categorical attributes: `city`, `road_type`, `weather`, `visibility`, `traffic_density`, `cause`, and `accident_severity`[cite: 1].
5. **Feature Scaling**:
   - Standardized all 41 feature columns using `StandardScaler` on train/test splits (`train_test_split(test_size=0.2, random_state=42)`)[cite: 1].

---

## 📌 Model Architecture
A sequential Feedforward Neural Network (Multi-Layer Perceptron) built using Keras[cite: 1]:

| Layer | Type | Activation | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| Input | InputLayer | — | (None, 41) | 0 |
| Layer 1 | Dense | ReLU | (None, 32) | 1,344 |
| Layer 2 | Dense | ReLU | (None, 16) | 528 |
| Layer 3 | Dense | ReLU | (None, 8) | 136 |
| Layer 4 | Dense | ReLU | (None, 4) | 36 |
| Layer 5 | Dense | ReLU | (None, 2) | 10 |
| Output | Dense | Linear | (None, 1) | 3 |
| **Total Parameters** | | | | **2,057 (8.04 KB)** |

---

## 📌 Training & Optimization
- **Loss Function**: Mean Squared Error (`mean_squared_error`)[cite: 1]
- **Optimizer**: Adam[cite: 1]
- **Regularization**: `EarlyStopping` callback monitoring `val_loss` (`min_delta=0.0001`, `patience=10`, `restore_best_weights=True`)[cite: 1]

---

## 📌 Installation & Usage

1. **Requirements**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow