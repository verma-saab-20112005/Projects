---
# Execution Results and Performance Metrics

This document outlines the output metrics, training history, and testing evaluation for the trained MNIST model.

---

## 📈 Training Summary (25 Epochs)

Training was completed over **25 epochs**[cite: 1]. Here are key highlights from the training process:

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Epoch 1** | 0.2698 | 92.13% | 0.1414 | 95.63% |
| **Epoch 5** | 0.0520 | 98.35% | 0.1017 | 97.22% |
| **Epoch 10** | 0.0223 | 99.24% | 0.1376 | 97.03% |
| **Epoch 15** | 0.0160 | 99.48% | 0.1266 | 97.56% |
| **Epoch 20** | 0.0102 | 99.66% | 0.1488 | 97.54% |
| **Epoch 25** | **0.0089** | **99.72%** | **0.1450** | **97.62%** |

*(Source logs extracted from notebook execution output)*[cite: 1]

---

## 📊 Loss Curves Analysis

The loss graph displays the progression over 25 epochs:
- **Blue Line:** Training Loss (Decreased steadily to ~0.0089)[cite: 1].
- **Green Line:** Validation Loss (Stabilized between 0.1006 and 0.1680)[cite: 1].

> **Observation:** The model converges quickly within 5–7 epochs, achieving near-perfect accuracy on the training set (~99.72%) with strong validation stability (~97.62%)[cite: 1].

---

## 🎯 Model Evaluation on Unseen Test Data

Using `sklearn.metrics.accuracy_score` on the **10,000 test images**:

```python
y_probability = model.predict(x_test_scaled)
y_pred = y_probability.argmax(axis=1)
accuracy_score(y_test, y_pred)
```[cite: 1]

- **Total Test Samples:** 10,000[cite: 1]
- **Final Test Accuracy:** **0.9791 (97.91%)**[cite: 1]