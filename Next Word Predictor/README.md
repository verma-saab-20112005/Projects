# FAQ & Maritime Regulation Text Generation using LSTMs

This repository contains two workflow implementations utilizing TensorFlow and Keras to train Recurrent Neural Networks (LSTMs) for next-word text prediction.

## Overview

The workflows demonstrate end-to-end text preprocessing and model generation:
1. **Dataset Tokenization**: Utilizing Keras `Tokenizer` to tokenize raw text data (DSMP course FAQ & Machinery/Pressure Vessel regulations).
2. **N-gram Sequence Creation**: Splitting text line-by-line and generating incremental n-gram sequences for language modeling.
3. **Padding & Data Splitting**: Standardizing sequence length using pre-padding (`pad_sequences`) and splitting data into training and testing sets via `train_test_split`.
4. **Target Encoding**: One-hot encoding categorical targets using `to_categorical`.
5. **Model Architecture**:
   - `Embedding` layer mapping target vocabulary size to continuous space.
   - `LSTM` recurrent layer to capture sequential dependencies.
   - `Dense` output layer with softmax activation for vocabulary-wide probability distribution.
6. **Inference & Text Generation**: Iterative loop predicting the next word based on seed text input.

## Requirements

- Python 3.10+
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-Learn

## How to Run

1. Open the Jupyter Notebook environment.
2. Run the preprocessing and tokenization cells.
3. Train the sequential model with categorical crossentropy loss.
4. Supply seed text to the inference loop to observe generative text continuation.