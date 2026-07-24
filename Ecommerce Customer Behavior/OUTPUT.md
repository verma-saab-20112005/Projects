---
# Analytical Findings & Exploratory Data Summary

## Data Summary & Distribution
- **Dataset Shape:** 50,000 rows × 25 columns[cite: 7].
- **Target Class Balance:**
  - **Retained (0):** 35,550 (71.1%)
  - **Churned (1):** 14,450 (28.9%)

### Summary Statistics Across Customer Classes (Mean Values)

| Variable | Retained Customers (0) | Churned Customers (1) | Key Difference / Observation |
| :--- | :--- | :--- | :--- |
| **Login Frequency** | 12.64 logins/mo | 9.12 logins/mo | Churned customers engage ~28% less frequently[cite: 7]. |
| **Customer Service Calls** | 5.19 calls | 6.90 calls | Churned customers make significantly higher support calls[cite: 7]. |
| **Days Since Last Purchase** | 26.89 days | 36.94 days | Inactivity window is 10 days higher for churned users[cite: 7]. |
| **Cart Abandonment Rate** | 54.19% | 64.18% | Churned users abandon carts at a 10% higher rate[cite: 7]. |
| **Email Open Rate** | 22.93% | 15.89% | Email engagement drops significantly prior to churn[cite: 7]. |
| **Average Order Value** | $118.38 | $134.76 | Churned customers have a higher average order value[cite: 7]. |
| **Credit Balance** | $2,088.59 | $1,664.20 | Unused credit balance is lower among churned customers[cite: 7]. |

## Identified Data Quality Anomalies
- **Outliers in Numerical Features:**
  - `Age` contains unrealistic maximum values up to 200 years[cite: 7].
  - `Total_Purchases` contains negative values down to -13[cite: 7].
  - `Cart_Abandonment_Rate` exceeds 100% with a maximum of 143.74%[cite: 7].
  - `Average_Order_Value` contains skewed outliers up to $9,666.38[cite: 7].
- **Missing Values:** Missing data exists across 14 numeric features (ranging from ~168 to 6,000 missing values).

## Strategic Insights & Action Items
1. **Support Intervention:** A proactive workflow should be triggered when customer service calls reach $\ge 6$ to address dissatisfaction before churn occurs[cite: 7].
2. **Re-engagement Triggers:** Automated promotional email sequences should target users exceeding 25–30 days since their last purchase[cite: 7].
3. **Cart Abandonment Optimization:** Targeted dynamic offers or retargeting banners should be deployed for users exceeding a 60% cart abandonment rate[cite: 7].