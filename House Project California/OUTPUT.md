---
# Model Prediction Output & Metrics Report

## Summary of Predictions
Predictions were computed on the test dataset (`Housing_Test_Data.xlsx`, 4,128 rows) using trained regression pipelines.

* **Primary Model Output**: `Housing_Test_Output_Data.xlsx` (Random Forest Regressor)
* **Secondary Model Output**: `Housing_Tset_Output_Data.xlsx` (Linear Regression Model)

---

## Model Cross-Validation Metrics Comparison

| Metric | Linear Regression RMSE ($) | Decision Tree RMSE ($) | Random Forest RMSE ($) |
| :--- | :--- | :--- | :--- |
| **Mean** | **$69,204.32** | **$69,081.36** | **$49,432.13** |
| **Std Dev** | $2,500.38 | $2,420.50 | $2,239.80 |
| **Min** | $65,318.22 | $64,770.56 | $45,940.43 |
| **25%** | $67,124.35 | $67,525.05 | $47,726.33 |
| **50% (Median)** | $69,404.66 | $69,027.99 | $49,230.48 |
| **75%** | $70,697.80 | $70,675.56 | $50,904.66 |
| **Max** | $73,003.75 | $73,280.39 | $53,301.09 |

---

## Sample Predicted Data (`Housing_Test_Output_Data.xlsx`)

| Row | Longitude | Latitude | Housing Median Age | Total Rooms | Total Bedrooms | Population | Households | Median Income | Ocean Proximity | Predicted Median House Value ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | -118.39 | 34.12 | 29 | 6447 | 1012.0 | 2184 | 960 | 8.2816 | `<1H OCEAN` | **$481,543.56** |
| 2 | -120.42 | 34.89 | 24 | 2020 | 307.0 | 855 | 283 | 5.0099 | `<1H OCEAN` | **$213,284.00** |
| 3 | -118.15 | 33.96 | 36 | 1857 | 400.0 | 1133 | 382 | 3.0156 | `<1H OCEAN` | **$178,450.12** |
| 4 | -118.11 | 34.15 | 40 | 1322 | 268.0 | 794 | 280 | 3.6382 | `<1H OCEAN` | **$242,100.05** |
| 5 | -121.93 | 37.03 | 12 | 3060 | 601.0 | 1184 | 480 | 5.1809 | `<1H OCEAN` | **$312,980.40** |

---

## Output Validation & Integrity Checks
1. **Row Count Match**: The test input (`Housing_Test_Data.xlsx`) and prediction files (`Housing_Test_Output_Data.xlsx`) both contain exactly **4,128 records**.
2. **Missing Values Handling**: Test data features with missing values in `total_bedrooms` were median-imputed via the scikit-learn preprocessing pipeline.
3. **Distribution Alignments**: Predicted values follow the real-world housing value distribution in California, with the Random Forest model providing the highest precision and lowest variance.