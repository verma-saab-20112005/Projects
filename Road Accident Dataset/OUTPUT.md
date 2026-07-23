---
# Execution Output & Data Summary

## 1. Initial Dataset Exploration
* **Total Rows**: 20,000[cite: 1]
* **Total Columns**: 24[cite: 1]
* **Duplicate Rows**: 0[cite: 1]

### Missing Values Summary:
* `festival`: 19,885 missing entries (Dropped)[cite: 1]
* All other 23 columns: 0 missing entries[cite: 1]

---

## 2. Categorical Value Distributions

### Visibility Distribution:
* `low`: 9,987[cite: 1]
* `high`: 6,690[cite: 1]
* `medium`: 3,323[cite: 1]

### Primary Causes Distribution:
* `distraction`: 4,026[cite: 1]
* `overspeeding`: 4,025[cite: 1]
* `weather`: 3,997[cite: 1]
* `drunk driving`: 3,978[cite: 1]
* `poor road`: 3,974[cite: 1]

### Accident Severity Distribution:
* `minor`: 11,025[cite: 1]
* `major`: 5,988[cite: 1]
* `fatal`: 2,987[cite: 1]

---

## 3. Data Split Summary
* **Total Samples**: 20,000[cite: 1]
* **Training Set (`x_train`)**: (16,000, 41)[cite: 1]
* **Testing Set (`x_test`)**: (4,000, 41)[cite: 1]

---

## 4. Keras Model Training Log

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 32)                1,344     
 dense_1 (Dense)             (None, 16)                528       
 dense_2 (Dense)             (None, 8)                 136       
 dense_3 (Dense)             (None, 4)                 36        
 dense_4 (Dense)             (None, 2)                 10        
 dense_5 (Dense)             (None, 1)                 3         
=================================================================
Total params: 2,057 (8.04 KB)
Trainable params: 2,057 (8.04 KB)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________

Training Progress:
Epoch 1/100 - 2s - loss: 0.0148 - accuracy: 0.0103 - val_loss: 0.0037 - val_accuracy: 0.0125
Epoch 2/100 - 0s - loss: 0.0022 - accuracy: 0.0112 - val_loss: 0.0015 - val_accuracy: 0.0125
Epoch 3/100 - 0s - loss: 0.0010 - accuracy: 0.0112 - val_loss: 8.5938e-04 - val_accuracy: 0.0125
Epoch 4/100 - 0s - loss: 6.3064e-04 - accuracy: 0.0112 - val_loss: 5.9930e-04 - val_accuracy: 0.0125
Epoch 5/100 - 0s - loss: 4.4124e-04 - accuracy: 0.0112 - val_loss: 4.2190e-04 - val_accuracy: 0.0125
... (Training continued with EarlyStopping regularization active)