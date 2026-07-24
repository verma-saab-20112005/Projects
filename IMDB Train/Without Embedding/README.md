# IMDB Sentiment Analysis using SimpleRNN

This project demonstrates binary sentiment classification on the IMDB Movie Reviews dataset using a Simple Recurrent Neural Network (SimpleRNN) in Keras and TensorFlow.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Workflow](#workflow)
- [Model Architecture](#model-architecture)
- [Training & Results](#training--results)

---

## Overview
The goal of this notebook is to process sequence data from the IMDB dataset (which contains movie reviews labeled as positive or negative) and train a `SimpleRNN` sequential network to classify the sentiment.

---

## Requirements

To run this notebook, install the following dependencies:

```bash
pip install pandas numpy tensorflow keras matplotlib

WorkflowData Loading:Downloads the preprocessed IMDB dataset using keras.datasets.imdb.Data Preprocessing:Evaluates text sequence length (e.g., target length set to 218).Applies pad_sequences with post-padding (padding="post", maxlen=218) to ensure uniform shapes across x_train and x_test.Datasets are reshaped to (25000, 218).Model Construction:Defines a Sequential model taking sequence inputs of shape (218, 1).Passes sequences through a SimpleRNN layer followed by fully connected Dense layers.Compilation & Fitting:Compiled using the Adam optimizer, binary_crossentropy loss, and accuracy metric.Trained for 25 epochs against validation data.Model ArchitectureLayer TypeConfiguration / Output ShapeActivationInput(218, 1)N/ASimpleRNN128 units (return_sequences=False)Default (tanh)Dense8 unitsReLUDense1 unitSigmoidTraining & ResultsEpochs: 25Optimizer: AdamLoss Function: Binary CrossentropyObservation: The raw sequence input without embedding layers leads to the model maintaining a baseline accuracy (~50.00%) across 25 epochs, indicating random guessing. Adding an Embedding layer is recommended for better feature representation.