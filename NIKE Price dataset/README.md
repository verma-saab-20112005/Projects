# Nike (NKE) Stock Volume Prediction Project

## Overview
This project processes historical stock data for Nike Inc. (NKE) spanning from January 2000 through January 2026. Using feature engineering and deep learning techniques with TensorFlow/Keras, the objective is to build a Sequential Neural Network model to predict trading volume based on price dynamics and temporal attributes.

## Features
* **Data Ingestion & Cleaning:** Reads raw stock historical data (`NKE.csv`) and checks for duplicated values.
* **Feature Engineering:** Converts `Date` into structured numerical components (`Year`, `Month`, `Day`).
* **Correlation Analysis:** Analyzes statistical relationships among variables (`Close`, `High`, `Low`, `Open`, `Volume`, `Year`, `Month`, `Day`).
* **Data Preprocessing:** Splits dataset into 80% training and 20% testing subsets, followed by `StandardScaler` feature normalization.
* **Deep Learning Model:** Constructs a multi-layer Neural Network with `ReLU` hidden activations and `Linear` output units trained using the `Adam` optimizer and `Mean Squared Error (MSE)` loss.

## Project Structure
```text
├── NKE.csv           # Historical Nike stock market dataset
├── notebook.ipynb    # Data exploration, preprocessing, and model training notebook
├── README.md         # Project documentation and summary
├── OUTPUT.md         # Execution outputs, evaluation details, and model summary
└── SUMMARY.html      # Graphical summary dashboard in web HTML format

Model Architecture
Input Layer: 7 features (Close, High, Low, Open, Year, Month, Day)

Dense Layer 1: 6 Neurons (ReLU activation)

Dense Layer 2: 4 Neurons (ReLU activation)

Dense Layer 3: 2 Neurons (ReLU activation)

Output Layer: 1 Neuron (Linear activation)

Total Parameters: 89 trainable parameters