# E-Commerce Customer Churn Analysis Project

## Overview
This repository contains an Exploratory Data Analysis (EDA) notebook designed to analyze customer behavior and churn drivers for an e-commerce platform[cite: 7]. The project inspects demographic characteristics, engagement metrics, purchase behavior, customer support interactions, and financial metrics to understand patterns associated with customer attrition.

## Dataset Profile
- **Dataset Name:** `ecommerce_customer_churn_dataset.csv`
- **Total Records:** 50,000 customers
- **Total Features:** 25 variables
- **Target Variable:** `Churned` (Binary: `0` = Retained, `1` = Churned)
- **Baseline Churn Rate:** 28.90% (14,450 churned vs. 35,550 retained)

## Key Features & Metadata
1. **Demographics:** `Age`, `Gender`, `Country`, `City`
2. **Account History:** `Membership_Years`, `Signup_Quarter`
3. **Engagement:** `Login_Frequency`, `Session_Duration_Avg`, `Pages_Per_Session`, `Email_Open_Rate`, `Social_Media_Engagement_Score`, `Mobile_App_Usage`
4. **Behavioral & Transactions:** `Cart_Abandonment_Rate`, `Wishlist_Items`, `Total_Purchases`, `Average_Order_Value`, `Days_Since_Last_Purchase`, `Discount_Usage_Rate`, `Returns_Rate`, `Product_Reviews_Written`
5. **Support & Financials:** `Customer_Service_Calls`, `Payment_Method_Diversity`, `Lifetime_Value`, `Credit_Balance`

## Dependencies
- Python 3.x
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

## Instructions for Execution
1. Ensure `ecommerce_customer_churn_dataset.csv` is present in the working directory.
2. Run the analysis notebook or script sequentially:
   ```bash
   jupyter notebook