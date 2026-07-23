---

# Output Summary

================================================================================
CUSTOMER CHURN MODEL - OUTPUT SUMMARY
DATASET PROFILE

• Total Dataset Records:  10,000 rows
• Feature Columns:        32 total columns
• Class Variable:         churn (Binary: 0 = Retained, 1 = Churned)
• Categorical Columns:    11 features (e.g., gender, country, city, customer_segment,
signup_channel, contract_type, complaint_type, survey_response)
• Numerical Columns:      21 features (e.g., age, tenure_months, monthly_logins,
avg_resolution_time, csat_score, email_open_rate)

PIPELINE EXECUTION SUMMARY

• Data Partitioning:      80/20 Stratified Train-Test Split

Training Set Size:    8,000 samples

Testing Set Size:     2,000 samples
• Preprocessing:

Categorical: Constant Imputation + One-Hot Encoding

Numerical: Median Imputation + Standard Scaling
• Model Architecture:     XGBoost Classifier (XGBClassifier)
• Optimization Method:    GridSearchCV (5-fold Cross-Validation)

MODEL EVALUATION

• Primary Evaluation Metric: Accuracy (%)
• Prediction Target Column: predicted_churn
• Output Integration:     Appended 'predicted_churn' predictions to the test set
DataFrame for error analysis and downstream validation.

================================================================================