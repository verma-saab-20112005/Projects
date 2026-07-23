---
# Execution Outputs & Analysis Report

## Dataset Overview
* **Dataset File:** `NKE.csv`
* **Total Records:** 6,559 rows × 6 initial columns
* **Target Variable:** `Volume`
* **Features Used:** `Close`, `High`, `Low`, `Open`, `Year`, `Month`, `Day`
* **Duplicates Found:** 0

---

## Dataset Info & Types
* **Memory Usage:** ~333.2 KB
* **Features Schema:**
  - `Close`: float64
  - `High`: float64
  - `Low`: float64
  - `Open`: float64
  - `Volume`: int64 (Target)
  - `Year`: int32
  - `Month`: int32
  - `Day`: int32

---

## Correlation Insights
From `data.corr()`, key relationships with `Volume` include:
* **Close / High / Low / Open:** Weak negative correlation with Volume (~ -0.25).
* **Year:** Negative correlation with Volume (~ -0.164), reflecting long-term changes in share liquidity and split adjustments.
* **Month / Day:** Negligible correlation with Volume (~ +0.008 and +0.073 respectively).

---

## Neural Network Model Summary

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 6)                 48        
 dense_1 (Dense)             (None, 4)                 28        
 dense_2 (Dense)             (None, 2)                 10        
 dense_3 (Dense)             (None, 1)                 3         
=================================================================
Total params: 89 (356.00 B)
Trainable params: 89 (356.00 B)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________
Training Overview
Optimizer: Adam

Loss Function: Mean Squared Error (mean_squared_error)

Metrics: Accuracy

Train / Test Split: 5,247 Training samples / 1,312 Testing samples (80/20 ratio)

Epochs Executed: 275 Epochs scheduled