---
# Data Quality & Null Value Audit Report

## Dataset Profile
* **Total Rows:** 2,524
* **Total Columns:** 13
* **Memory Usage:** ~256.5 KB

## Missing Value Diagnostics

The null value profile reflects expected behavior for time-series rolling calculations:

```text
Date                0
Adj Close           0
Close               0
High                0
Low                 0
Open                0
Volume              0
MA_50              49
MA_200            199
Daily_Return        1
Year                0
Month               0
Volatility_30d     30
dtype: int64
Explanatory Findings
MA_50 (49 Missing Values):

Root Cause: Calculation of a 50-day Simple Moving Average requires a minimum window of 50 data points, leaving indices 0 through 48 empty.

MA_200 (199 Missing Values):

Root Cause: A 200-day Simple Moving Average requires a minimum window of 200 data points, leaving indices 0 through 198 empty.

Volatility_30d (30 Missing Values):

Root Cause: 30-day rolling volatility calculation requires prior 30 window records.

Daily_Return (1 Missing Value):

Root Cause: The initial row in the time series (t_0) lacks a preceding reference point (t_-1) to derive a daily return percentage.

Data Integrity Verification
Core Market Indicators: (Open, High, Low, Close, Adj Close, Volume) have 100% data completeness (0 missing records across all 2,524 trading sessions).