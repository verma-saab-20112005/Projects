# Delhi Air Quality Index (AQI) 2025 Analysis

## 📌 Project Overview
This project presents an exploratory data analysis (EDA) of air quality and meteorological metrics across Delhi for the year 2025. Using Python data science libraries, the analysis inspects dataset structures, missing values, column data types, and hourly pollutant/weather indicators.

## 📁 Dataset Details
* **Source File**: `Delhi_AQI_2025.xlsx`
* **Total Records**: 52,560 rows
* **Total Columns**: 16 columns
* **Missing Values**: 0 (Clean dataset)

### Features / Columns Included
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `date_ist` | object | Date in Indian Standard Time (IST) |
| `time_ist` | object | Time of observation in IST |
| `location` | object | Station/Area location (e.g., Anand Vihar) |
| `lat` | float64 | Latitude coordinates |
| `lon` | float64 | Longitude coordinates |
| `temp_c` | float64 | Temperature in Celsius |
| `humidity` | int64 | Relative Humidity percentage |
| `pressure_mb` | float64 | Atmospheric Pressure in millibars |
| `windspeed_kph` | float64 | Wind speed in km/h |
| `condition_text` | object | Weather condition description |
| `description` | object | WMO weather code description |
| `aqi_index` | int64 | Calculated Air Quality Index |
| `pm2_5` | float64 | Particulate Matter 2.5 concentration ($\mu g/m^3$) |
| `pm10` | float64 | Particulate Matter 10 concentration ($\mu g/m^3$) |
| `co` | int64 | Carbon Monoxide concentration |
| `no2` | float64 | Nitrogen Dioxide concentration |

## 🛠️ Environment & Requirements
To run the notebook, ensure you have Python installed along with the following packages:

```bash
pip install pandas numpy openpyxl matplotlib seaborn