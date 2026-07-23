# Gaming Performance & Productivity Analysis

## Project Overview
This repository contains an exploratory data analysis (EDA) of a dataset (`Performance.csv`) comprising 1,000 gaming enthusiasts, students, and working professionals. The analysis evaluates how daily/weekly gaming habits, game genres, sleep patterns, stress levels, and focus levels impact overall productivity, academic or work scores, and general performance.

## File Structure
* `notebook.ipynb`: Jupyter notebook containing data loading, preliminary inspection, descriptive statistics, frequency distributions, data type auditing, and visual distribution plots.
* `Performance.csv`: Primary dataset containing demographic and gaming habits data.
* `README.md`: Project documentation and dataset overview.
* `OUTPUT.md`: Concise analytical findings and statistical output derived from the notebook execution.
* `SUMMARY.html`: Web-based executive summary report featuring responsive styling and key metrics.

## Key Dataset Variables
* **User_ID**: Unique identifier for each participant (U0001 to U1000).
* **Age**: Participant age (range: 18 - 35 years).
* **Gender**: Male / Female.
* **Occupation**: Student / Working Professional.
* **Game_Type**: Action, Casual, Puzzle, Simulation, Sports, Strategy.
* **Daily_Gaming_Hours**: Average daily hours spent gaming (0.5 - 6.0 hours).
* **Weekly_Gaming_Hours**: Total weekly hours spent gaming (3.5 - 42.0 hours).
* **Primary_Gaming_Time**: Morning, Evening, Night.
* **Sleep_Hours**: Average nightly sleep duration (4.5 - 8.5 hours).
* **Stress_Level**: Self-reported stress index (Scale: 2 to 9).
* **Focus_Level**: Self-reported focus index (Scale: 3 to 9).
* **Academic_or_Work_Score**: Evaluated academic/work performance score (Scale: 55 to 95).
* **Productivity_Level**: Self-reported productivity score (Scale: 50 to 100).
* **Performance_Impact**: Categorical impact assessment (Neutral, Negative, Positive).

## Requirements & Usage
To run the analysis notebook locally:

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook