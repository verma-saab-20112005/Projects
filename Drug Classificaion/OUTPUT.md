---
# Execution & Model Evaluation Output Report

## 1. Input Data Profile & Summary Statistics

### Data Structure (`data.info()`)
- **Total Entries**: 200
- **Total Columns**: 6
- **Missing Values**: None (200 non-null per column)

| Column Name | Non-Null Count | Data Type | Notes |
| :--- | :--- | :--- | :--- |
| `Age` | 200 | `int64` | Patient Age |
| `Sex` | 200 | `object` | `F` or `M` |
| `BP` | 200 | `object` | `HIGH`, `LOW`, `NORMAL` |
| `Cholesterol` | 200 | `object` | `HIGH`, `NORMAL` |
| `Na_to_K` | 200 | `float64` | Sodium to Potassium ratio |
| `Drug` | 200 | `object` | Target Drug Name |

---

## 2. Target Variable Analysis (`Drug`)

```text
Drug Class Counts:
DrugY    91
drugX    54
drugA    23
drugC    16
drugB    16
Name: count, dtype: int64
3. Data Preprocessing & Splitting OutputsFeature Matrix Shapes:x original: (200, 5)x post One-Hot Encoding: (200, 9)Dataset Splitting:Training target shape (y_train): (160,)Validation/Test target shape (y_test): (40,)Training One-Hot Target Matrix (y_train_encoded): (160, 5)4. Model Training Log & Metrics ProgressionEarly Stopping ConfigurationMonitored Metric: val_lossPatience: 5 epochsMinimum Delta: 0.001Total Epochs Executed: 123 / 150Key Training MilestonesEpoch StageTrain LossTrain AccuracyVal LossVal AccuracyNotesEpoch 11.733128.91%1.718825.00%Initial Model StateEpoch 250.750473.44%0.768971.88%Steady ConvergenceEpoch 500.519181.25%0.390781.25%Loss below 0.40Epoch 750.365087.50%0.200793.75%Val Accuracy hits 93.75%Epoch 1070.315585.94%0.1243100.00%Peak Val AccuracyEpoch 1180.335584.38%0.113596.88%Minimum Val LossEpoch 1230.379582.81%0.114196.88%Early Stopping Triggered5. Summary FindingsHigh Convergence Rate: The model converged rapidly, reaching over 90% validation accuracy by Epoch 59 (val_acc = 90.62%).Effective Regularization: The inclusion of a 30% Dropout layer (Dropout(0.3)) stabilized validation metrics and prevented severe overfitting despite the small dataset size (200 records).High Predictive Performance: Reached a minimum validation loss of 0.1135 and peak validation accuracy of 100%.