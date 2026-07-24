---
# Data Analysis Output & Summary Statistics

## Dataset Summary Statistics
Below is the descriptive statistical summary generated for the numerical features across all 18,270 records:

| Statistic | Open ($) | High ($) | Low ($) | Close ($) | Volume | Daily Change (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Count** | 18,270 | 18,270 | 18,270 | 18,270 | 18,270 | 18,270 |
| **Mean** | 20,442.81 | 20,718.69 | 20,166.62 | 20,442.38 | 25,608,340 | 0.0060% |
| **Std Dev** | 11,257.36 | 11,260.18 | 11,258.55 | 11,261.45 | 14,148,370 | 4.5341% |
| **Min** | 1,000.65 | 1,018.52 | 349.96 | 622.95 | 1,005,194 | -44.32% |
| **25% (Q1)**| 10,743.36 | 11,011.50 | 10,471.05 | 10,736.96 | 13,365,300 | -1.25% |
| **50% (Med)**| 20,398.86 | 20,685.67 | 20,116.40 | 20,394.74 | 25,737,160 | -0.01% |
| **75% (Q3)**| 30,208.95 | 30,470.58 | 29,938.70 | 30,186.75 | 37,783,440 | 1.23% |
| **Max** | 39,990.31 | 40,680.18 | 39,985.49 | 40,474.78 | 49,998,020 | 45.69% |

## Data Quality Checks
- **Missing Values**: 0 null entries found across all 9 original columns.
- **Memory Usage**: ~1.3+ MB.
- **Data Types**: `Date` successfully converted to datetime format before feature extraction.

## Outputs & Visualizations
- Visualizations were rendered for all key metrics (`Open`, `High`, `Low`, `Close`, `Volume`, `Daily_Change_Percent`, `Year`, `Month`, `Day`).