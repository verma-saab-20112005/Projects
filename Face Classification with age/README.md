# Face Classification with Age & Gender (UTKFace Dataset)

This repository contains data preprocessing and dataset preparation routines for age and gender classification using the **UTKFace** dataset. The project utilizes TensorFlow, Keras, and Pandas for processing facial images.

> **Note:** The underlying image database (UTKFace) is not included in this repository due to file size constraints. Instructions to download and structure the dataset locally are provided below.

---

## 🛠️ Requirements & Setup

### Prerequisites
Make sure you have Python installed along with the following libraries:

```bash
pip install pandas numpy tensorflow keras

Dataset Overview & Setup
The project uses the UTKFace dataset, which consists of face images labeled with age, gender, and ethnicity.

Filename Format
UTKFace images are named using the following format:
[age]_[gender]_[race]_[date&time].jpg

age: An integer indicating the age (ranging from 1 to 116 years).

gender:

0: Male

1: Female

race: An integer from 0 to 4 (White, Black, Asian, Indian, Others).

date&time: Unique timestamp recorded for the cropped image file.

Local Setup Instructions
Download the UTKFace dataset from its official source or Kaggle.

Extract the dataset folder into your target directory.

Update the folder path inside your script or notebook:

How It Works
Imports Modules: Loads necessary data processing (pandas, numpy, os) and deep learning (tensorflow, keras) libraries.

Directory Parsing: Scans all image filenames within the dataset directory using os.listdir().

Metadata Extraction: Splits each filename by the _ delimiter to extract target variables (age and gender).

Data Verification: Extracts distinct age sets and sample gender label distributions for dataset analysis.
