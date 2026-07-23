---
# Execution Output & Data Summary

## Notebook Execution Logs

### Cell 1–3: Dependencies & Data Ingestion
- Imports: `pandas`, `numpy`
- Loaded Dataset: `train.xlsx`
- Column Count: **9**

### Cell 7–8: Type Conversion & Inspection
Categorical text columns converted to `string` dtypes. Column data types verified as follows:

| Column Index | Data Type | Inferred Variable |
| :--- | :--- | :--- |
| 0 | `int64` | Store ID |
| 1 | `int64` | DayOfWeek |
| 2 | `datetime64[ns]` | Date |
| 3 | `int64` | Sales |
| 4 | `int64` | Customers |
| 5 | `int64` | Open |
| 6 | `int64` | Promo |
| 7 | `string` | StateHoliday / Type |
| 8 | `int64` | SchoolHoliday |

### Cell 9: Data Visualization Output
A 3x3 subplot grid was generated to inspect variable distributions across all 9 attributes:
- **Grid Layout**: 3 rows $\times$ 3 columns
- **Axes Visualized**: `Store`, `DayOfWeek`, `Date`, `Sales`, `Customers`, `Open`, `Promo`, `SchoolHoliday`, and categorical text flags.
- **Output Rendered**: Multi-panel exploratory plot inline.