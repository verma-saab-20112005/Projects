---
# Model Execution Output and Training Results

## Dataset Summary
- **Training Set**: 791 images (2 classes)
- **Testing Set**: 209 images (2 classes)

---

## Model Architecture Summary Output

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                    Output Shape           Param #   
=================================================================
 conv2d (Conv2D)                 (None, 254, 254, 32)   896       
 batch_normalization             (None, 254, 254, 32)   128       
 max_pooling2d (MaxPooling2D)    (None, 127, 127, 32)   0         
 conv2d_1 (Conv2D)               (None, 125, 125, 64)   18496     
 batch_normalization_1           (None, 125, 125, 64)   256       
 max_pooling2d_1 (MaxPooling2D)  (None, 62, 62, 64)     0         
 conv2d_2 (Conv2D)               (None, 60, 60, 128)    73856     
 batch_normalization_2           (None, 60, 60, 128)    512       
 max_pooling2d_2 (MaxPooling2D)  (None, 30, 30, 128)    0         
 flatten (Flatten)               (None, 115200)         0         
 dense (Dense)                   (None, 128)            14745728  
 dropout (Dropout)               (None, 128)            0         
 dense_1 (Dense)                 (None, 64)             8256      
 dense_2 (Dense)                 (None, 32)             2080      
 dense_3 (Dense)                 (None, 16)             528       
 dense_4 (Dense)                 (None, 8)              136       
 dense_5 (Dense)                 (None, 4)              36        
 dropout_1 (Dropout)             (None, 4)              0         
 dense_6 (Dense)                 (None, 1)              5         
=================================================================
Total params: 14,850,913 (56.65 MB)
Trainable params: 14,850,465 (56.65 MB)
Non-trainable params: 448 (1.75 KB)
_________________________________________________________________
Epoch Execution LogTraining was terminated early at epoch 18 due to the Early Stopping callback monitoring val_loss.EpochTraining LossTraining AccuracyValidation LossValidation Accuracy11.371152.47%0.687657.42%20.743457.90%0.952455.98%30.634563.97%0.686155.02%40.564669.91%0.686048.33%50.603268.77%0.640751.20%60.537577.62%0.573470.33%70.554374.84%0.572868.90%80.408681.80%0.580175.12%90.367380.40%0.489578.47%100.340987.99%0.543478.47%110.363882.55%0.607477.03%120.282988.37%0.688980.38%130.267190.64%0.416986.12%140.204393.17%0.628179.90%150.150193.43%0.631588.04%160.128895.70%3.234269.86%170.100895.95%2.144071.29%180.144494.82%0.895488.52%Best Performance: Epoch 13 achieved the lowest validation loss of 0.4169 with a validation accuracy of 86.12%.