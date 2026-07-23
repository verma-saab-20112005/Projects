# MNIST Digit Classification using Keras Sequential Neural Network

This repository contains a Deep Learning project built with TensorFlow/Keras to classify handwritten digits (0–9) using the classic **MNIST dataset**.

## 📌 Project Overview
- **Dataset:** MNIST Handwritten Digit Database (60,000 training images, 10,000 testing images)[cite: 1]
- **Frameworks & Libraries:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn[cite: 1]
- **Model Type:** Multilayer Perceptron (Sequential Feedforward Neural Network)[cite: 1]
- **Final Test Accuracy:** **~97.91%**[cite: 1]

---

## 🛠️ Data Preprocessing
1. **Normalization:** The pixel values range from 0 to 255[cite: 1]. They are scaled down to the range `[0, 1]` by dividing by `255.0`:
   ```python
   x_train_scaled = x_train / 255.0
   x_test_scaled = x_test / 255.0
   ```[cite: 1]

---

## 🏗️ Neural Network Architecture

The model is built using `keras.Sequential` with dense layers:

```python
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(10, activation="softmax")
])
```[cite: 1]

### Model Details:
- **Input Shape:** `(28, 28)` 2D image arrays[cite: 1]
- **Flatten Layer:** Converts 2D 28x28 images into a 1D vector of size 784[cite: 1].
- **Hidden Layer 1:** 128 neurons, `ReLU` activation[cite: 1].
- **Hidden Layer 2:** 64 neurons, `ReLU` activation[cite: 1].
- **Hidden Layer 3:** 32 neurons, `ReLU` activation[cite: 1].
- **Output Layer:** 10 neurons, `Softmax` activation for multi-class probability distribution[cite: 1].

---

## ⚙️ Compilation & Training Configuration

- **Optimizer:** Adam[cite: 1]
- **Loss Function:** `sparse_categorical_crossentropy`[cite: 1]
- **Metrics:** Accuracy[cite: 1]
- **Epochs:** 25[cite: 1]
- **Validation Split:** 20% (12,000 samples used for validation, 48,000 for training)[cite: 1]

```python
model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(x_train_scaled, y_train, epochs=25, validation_split=0.2, verbose=1)
```[cite: 1]

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn