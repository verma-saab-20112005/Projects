# Global Stock Market Data Analysis (2020–2024)

## Overview
This repository contains an exploratory data analysis (EDA) pipeline implemented in Python via Jupyter Notebook. The project processes and cleans a global stock market dataset spanning from **January 1, 2020 to December 31, 2024** across 9 major financial market indices globally.

## Dataset Structure
The dataset (`Stock_data.csv`) contains **18,270 records** with no missing values. 

### Columns Overview:
- `Date`: Date of stock trade recording (`datetime64[ns]`)
- `Index_Name`: Name of the stock index (e.g., S&P 500, NASDAQ Composite, Dow Jones, FTSE 100, Nikkei 225, Hang Seng, DAX, CAC 40, SSE Composite, KSE 100)
- `Country`: Country/Region of origin
- `Open`: Opening price (`float64`)
- `High`: Highest price of the trading session (`float64`)
- `Low`: Lowest price of the trading session (`float64`)
- `Close`: Closing price (`float64`)
- `Volume`: Total volume traded (`int64`)
- `Daily_Change_Percent`: Daily percentage change in price (`float64`)

## Feature Engineering
The notebook extracts temporal features from the `Date` column for granular trend analysis:
- `Year`: Year extracted (`2020` to `2024`)
- `Month`: Month extracted (`1` to `12`)
- `Day`: Day extracted (`1` to `31`)

## Execution & Requirements
To run this notebook, ensure you have Python 3.x installed along with the following libraries:
```bash
pip install pandas numpy matplotlib seaborn