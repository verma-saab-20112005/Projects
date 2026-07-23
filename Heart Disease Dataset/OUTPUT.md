# Execution Output & Results Summary

This document details the execution output, model parameter counts, loss/accuracy curves, and training metrics for the Heart Disease Classification model.

---

## 1. Dataset Breakdown & Info

- **Shape:** 1,025 rows, 14 columns (13 features + 1 target)
- **Missing Values:** 0 non-null values across all columns
- **Memory Usage:** ~104.2 KB

### Target Class Distribution
```text
1 (Heart Disease):     526 patients (51.32%)
0 (No Heart Disease):  499 patients (48.68%)
Total:                 1,025 patients

Data SplitsTotal Dataset: 1,025 samplesTraining Set (80%): 820 samplesTest Set (20%): 205 samplesValidation Split (20% of Training Set): 164 validation / 656 sub-training samples2. Keras Model Architecture SummaryPlaintextModel: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 8)                 112       
 dense_1 (Dense)             (None, 4)                 36        
 dropout (Dropout)           (None, 4)                 0         
 dense_2 (Dense)             (None, 2)                 10        
 dense_3 (Dense)             (None, 1)                 3         
=================================================================
Total params: 161 (644.00 B)
Trainable params: 161 (644.00 B)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________
3. Training Epoch History Summary (Key Milestones)EpochTrain LossTrain AccuracyVal LossVal AccuracyNotes10.681356.71%0.691649.39%Initial epoch100.479278.51%0.527675.00%Rapid convergence250.378484.30%0.463782.32%Steady improvement500.339186.89%0.438485.98%Accuracy stabilizing750.299788.11%0.408086.59%High validation performance840.279690.24%0.398587.20%Best Weights Restored890.277790.24%0.398487.20%Early stopping triggered4. Key Performance IndicatorsOptimal Epoch: 84Final Validation Loss: 0.3985Final Validation Accuracy: 87.20%Training Set Accuracy: 90.24%Training/Validation Gap: ~3.04% (indicating controlled generalization with low overfitting)