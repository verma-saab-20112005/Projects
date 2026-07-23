# Weather History Analysis Project

## Overview
This repository contains a exploratory data analysis (EDA) pipeline executed on a 4-year historical weather dataset (`Weather_history.xlsx`). The dataset tracks daily weather metrics—including daily precipitation, temperature range (min/max), wind speed, and weather condition classifications—from **January 1, 2012** to **December 31, 2015** (1,461 daily records).

---

## Dataset Schema

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `date` | `datetime64[ns]` | Observation date (YYYY-MM-DD) |
| `precipitation` | `float64` | Daily precipitation measurement (mm) |
| `temp_max` | `float64` | Maximum daily temperature (°C) |
| `temp_min` | `float64` | Minimum daily temperature (°C) |
| `wind` | `float64` | Average daily wind speed (m/s) |
| `weather` | `string` / `object` | General weather condition (`rain`, `sun`, `fog`, `drizzle`, `snow`) |

---

## Code Execution Pipeline

The Jupyter Notebook executes the following workflow:

1. **Data Loading:**
   Loads the dataset from `Weather_history.xlsx` into a Pandas Dataframe.
   ```python
   import numpy as np
   import pandas as pd

   data = pd.read_excel('Weather_history.xlsx')

   **Categorical & Statistical Exploration:**
   - Performs frequency counts on the `weather` column (`data["weather"].value_counts()`).
   - Generates summary statistics across numeric variables (`data.describe()`).
   - Produces EDA histograms for distribution analysis across variables.

---

## Requirements

- Python 3.8+
- Pandas
- NumPy
- Matplotlib / Seaborn
- OpenPyXL (for reading `.xlsx` files)

---

## Key Insights

- **Rain** and **Sun** dominate the climate, representing **87.7%** of all recorded days over the 4-year period.
- Maximum temperatures reach up to **35.6°C**, while minimum temperatures drop as low as **-7.1°C**.
- Extreme precipitation events occur up to **55.9 mm** in a single day, while the average daily rainfall is **3.03 mm**.