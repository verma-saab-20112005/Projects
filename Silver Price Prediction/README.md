# Silver Market Data Analytics Pipeline

A Python/Pandas data analytics pipeline designed to process, validate, and extract quantitative technical indicators from historical silver market data (`silver_prices_data.xlsx`).

## Project Overview

This repository analyzes daily financial time-series metrics for silver, computing moving averages, daily percentage returns, and 30-day volatility metrics across 2,524 trading records.

## Data Schema

The raw dataset (`silver_prices_data.xlsx`) comprises **2,524 entries** and **13 features**:

| Column Name | Data Type | Non-Null Count | Description |
| :--- | :--- | :--- | :--- |
| `Date` | `datetime64[ns]` | 2,524 | Date of trading session |
| `Adj Close` | `float64` | 2,524 | Adjusted closing price |
| `Close` | `float64` | 2,524 | Unadjusted closing price |
| `High` | `float64` | 2,524 | Session high price |
| `Low` | `float64` | 2,524 | Session low price |
| `Open` | `float64` | 2,524 | Session opening price |
| `Volume` | `int64` | 2,524 | Total trading volume |
| `MA_50` | `float64` | 2,475 | 50-day simple moving average |
| `MA_200` | `float64` | 2,325 | 200-day simple moving average |
| `Daily_Return` | `float64` | 2,523 | Day-over-day price return |
| `Year` | `int64` | 2,524 | Calendar year |
| `Month` | `int64` | 2,524 | Calendar month |
| `Volatility_30d` | `float64` | 2,494 | 30-day rolling price volatility |

## Prerequisites

* Python 3.8+
* `pandas`
* `numpy`
* `openpyxl`

## Execution Instructions

```bash
# Clone repository and navigate to root
git clone <repo_url>
cd silver-analysis

# Install dependencies
pip install pandas numpy openpyxl

# Execute analysis notebook
jupyter notebook silver_analysis.ipynb