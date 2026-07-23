---

### File 2: `OUTPUT.md`

```markdown
# Training Results & Performance Output

## Executive Summary
The Keras deep learning model achieved strong convergence and high classification accuracy within **34 training epochs** before Early Stopping triggered.

---

## Key Training Metrics

| Metric | Epoch 1 | Epoch 17 (Peak Val) | Epoch 34 (Final) |
| :--- | :--- | :--- | :--- |
| **Training Accuracy** | 85.33% | 96.89% | **97.24%** |
| **Training Loss** | 0.4679 | 0.1637 | **0.1519** |
| **Validation Accuracy** | 90.08% | 96.43% | **96.28%** |
| **Validation Loss** | 0.3350 | 0.1695 | **0.1696** |

- **Best Validation Accuracy**: **96.86%** (Epoch 29)
- **Best Validation Loss**: **0.1565** (Epoch 29)
- **Stopping Trigger**: Early stopping triggered at Epoch 34 due to `val_loss` stabilization.

---

## Data Breakdown Summary

- **Total Rows**: 52,471 laps
- **Non-PitStop Laps (0)**: 42,627
- **PitStop Laps (1)**: 9,844
- **Expanded Feature Count**: 67 features

### Top Drivers by Dataset Laps
1. Lando Norris (`NOR`): 2,792
2. Max Verstappen (`VER`): 2,781
3. George Russell (`RUS`): 2,758
4. Fernando Alonso (`ALO`): 2,755
5. Oscar Piastri (`PIA`): 2,709

### Compound Breakdown
- **HARD**: 25,880
- **MEDIUM**: 18,090
- **SOFT**: 5,588
- **INTERMEDIATE**: 2,834
- **WET**: 79