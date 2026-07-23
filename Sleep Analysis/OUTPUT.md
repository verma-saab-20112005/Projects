# Model Architecture & Data Summary Output

## Dataset Distribution Summary

### Target Class Distribution (`Sleep Disorder`)
- **None**: 219 instances
- **Sleep Apnea**: 78 instances
- **Insomnia**: 77 instances

### Top Occupations Count
- **Nurse**: 73
- **Doctor**: 71
- **Engineer**: 63
- **Lawyer**: 47
- **Teacher**: 40
- **Accountant**: 37
- **Salesperson**: 32
- **Scientist**: 4
- **Software Engineer**: 4
- **Sales Representative**: 2
- **Manager**: 1

### Gender Distribution
- **Male**: 189
- **Female**: 185

---

## Keras Sequential Model Summary

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 16)                432       
 dropout (Dropout)           (None, 16)                0         
 dense_1 (Dense)             (None, 8)                 136       
 dropout_1 (Dropout)         (None, 8)                 0         
 dense_2 (Dense)             (None, 4)                 36        
 dropout_2 (Dropout)         (None, 4)                 0         
 dense_3 (Dense)             (None, 3)                 15        
=================================================================
Total params: 619 (2.42 KB)
Trainable params: 619 (2.42 KB)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________

Data Transformation Shapes
Raw Feature Columns: 13

Processed Feature Columns (x): 26 (after One-Hot Encoding and Blood Pressure splitting)

Training Set (x_train): 299 samples, 26 features

Testing Set (x_test): 75 samples, 26 features