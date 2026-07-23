# Store Sales Analysis & Data Processing Pipeline

## Project Overview
This project provides an automated data processing and analytical workspace for large-scale time-series retail datasets (specifically formatted for store sales and performance tracking). The notebook cleans column data types, parses date components, and summarizes key retail performance metrics across stores, dates, custom promotional events, and customer metrics.

## Features
- **Data Ingestion & Typing**: Efficiently reads large Excel files (`train.xlsx`) and casts string/categorical features explicitly.
- **Retail Metric Tracking**: Prepares features for sales volume, customer counts, open/closed store status, promotional periods (`Promo`), and holidays (`SchoolHoliday`).
- **Exploratory Visualizations**: Visualizes store performance distribution across day-of-week, promotion impacts, and customer traffic.

## Dataset Structure
The project is built to handle retail store sales data containing the following 9 primary attributes:
1. `Store`: Unique Store Identifier (Integer)
2. `DayOfWeek`: Day of the week 1-7 (Integer)
3. `Date`: Entry Date (Datetime)
4. `Sales`: Total turnover for the given day (Integer)
5. `Customers`: Total number of customers on that day (Integer)
6. `Open`: Indicator for whether the store was open (1) or closed (0)
7. `Promo`: Indicator for whether a store was running a promotion on that day (1/0)
8. `StateHoliday` / `SchoolHoliday`: Holiday categorizations (String/Integer)
9. Additional store attributes or secondary flags.

> **Note on Dataset Access**: The full dataset (`train.xlsx`) exceeds standard upload limits and is kept externally in local/cloud storage. Place your dataset in the root directory prior to running the notebook.

## Getting Started

### Prerequisites
- Python 3.8+
- Pandas
- NumPy
- Matplotlib / Seaborn
- OpenPyXL (for Excel parsing)

### Setup & Execution
1. Clone this repository.
2. Ensure `train.xlsx` is placed in the project root directory.
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook