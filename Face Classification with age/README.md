# Face Classification and Age Extraction (UTKFace)

This project focuses on processing and preparing the **UTKFace** dataset for multi-task deep learning applications, specifically estimating age and classifying gender from facial image data using TensorFlow and Keras.

## Project Overview
The notebook extracts target metadata (age and gender) embedded in the filenames of the UTKFace dataset to construct label sets suitable for training neural network models.

- **Age Dataset Range**: Extracted age values range from **1 to 116 years old**.
- **Gender Classes**: Binary classification (0 and 1).
- **Frameworks Used**: TensorFlow, Keras, Pandas, NumPy.

---

## Dataset Notice
> **Note**: Due to storage limits and the large size of the original dataset, the **UTKFace image folder is not included** in this repository.

### Filename Format
The dataset relies on UTKFace's naming convention:
`[age]_[gender]_[race]_[date&time].jpg`

Example: `100_0_0_20170112213500903.jpg.chip.jpg`
- **Age**: `100`
- **Gender**: `0` (Male/Female depending on schema)
- **Race**: `0`

---
Code Structure
Imports: Imports TensorFlow, Keras ImageDataGenerator, NumPy, Pandas, and OS modules.

Path Parsing: Iterates over files in the dataset folder and splits filenames by _.

Data Extraction:

age: Extracted from index 0.

gender: Extracted from index 1.

img_path: Extracted full filename.