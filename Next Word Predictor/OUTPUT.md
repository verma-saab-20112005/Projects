# Execution Output & Model Training Logs

## Data Preprocessing Summary

### Notebook 1 (DSMP FAQ Corpus)
- **Total Vocabulary Size**: 282 unique words
- **Total Documents Processed**: 1
- **Total Generated Sequence Samples**: 863
- **Max Sequence Length (Padded)**: 57
- **Training Feature Matrix (`x_train`)**: Shape `(690, 56)`
- **Testing Feature Matrix (`x_test`)**: Shape `(173, 56)`
- **One-Hot Target Shape (`y_train`, `y_test`)**: `(690, 283)` and `(173, 283)`

## Model Architecture (`model.summary()`)

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 56, 100)           28,300    
 lstm (LSTM)                 (None, 150)               150,600   
 dense (Dense)               (None, 283)               42,733    
=================================================================
Total params: 221,633 (865.75 KB)
Trainable params: 221,633 (865.75 KB)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________

Model Training Output Snippet
Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 100

Plaintext
Epoch 1/100  - 22/22 [==================] - 0s 16ms/step - accuracy: 0.9449 - loss: 0.1359
Epoch 50/100 - 22/22 [==================] - 0s 15ms/step - accuracy: 0.9420 - loss: 0.1160
Epoch 100/100- 22/22 [==================] - 0s 15ms/step - accuracy: 0.9449 - loss: 0.1115
Inference Output
Seed Input Text: "What is the course fee"

Generated Iterative Sequence:

What is the course fee for

What is the course fee for monthly

What is the course fee for monthly subscription

What is the course fee for monthly subscription suppose

What is the course fee for monthly subscription suppose if

What is the course fee for monthly subscription suppose if you

What is the course fee for monthly subscription suppose if you pay

What is the course fee for monthly subscription suppose if you pay on

What is the course fee for monthly subscription suppose if you pay on 15th

What is the course fee for monthly subscription suppose if you pay on 15th jan