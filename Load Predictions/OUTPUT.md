---
# Analysis Output & Findings

## Dataset Structure
* **Total Rows:** 20,000[cite: 19]
* **Total Columns:** 22[cite: 19]

### Column Data Types
* **Numeric (`int64` / `float64`):** `age`, `annual_income`, `monthly_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`, `loan_term`, `installment`, `num_of_open_accounts`, `total_credit_limit`, `current_balance`, `delinquency_history`, `public_records`, `num_of_delinquencies`, `loan_paid_back`[cite: 19].
* **Categorical (`object`):** `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`[cite: 19].

---

## Categorical Feature Value Counts

### 1. Gender Distribution
* **Female:** 10,034 (50.17%)[cite: 19]
* **Male:** 9,536 (47.68%)[cite: 19]
* **Other:** 430 (2.15%)[cite: 19]

### 2. Marital Status
* **Single:** 9,031 (45.16%)[cite: 19]
* **Married:** 8,974 (44.87%)[cite: 19]
* **Divorced:** 1,428 (7.14%)[cite: 19]
* **Widowed:** 567 (2.84%)[cite: 19]

### 3. Education Level
* **Bachelor's:** 8,045 (40.23%)[cite: 19]
* **High School:** 5,919 (29.60%)[cite: 19]
* **Master's:** 3,724 (18.62%)[cite: 19]
* **Other:** 1,508 (7.54%)[cite: 19]
* **PhD:** 804 (4.02%)[cite: 19]

### 4. Employment Status
* **Employed:** 13,007 (65.04%)[cite: 19]
* **Self-employed:** 2,923 (14.62%)[cite: 19]
* **Unemployed:** 2,113 (10.57%)[cite: 19]
* **Retired:** 1,176 (5.88%)[cite: 19]
* **Student:** 781 (3.91%)[cite: 19]

### 5. Primary Loan Purpose
* **Debt Consolidation:** 7,981 (39.91%)[cite: 19]
* **Other:** 2,550 (12.75%)[cite: 19]
* **Car:** 2,390 (11.95%)[cite: 19]
* **Home:** 1,972 (9.86%)[cite: 19]
* **Education:** 1,675 (8.38%)[cite: 19]
* **Business:** 1,629 (8.15%)[cite: 19]
* **Medical:** 1,196 (5.98%)[cite: 19]
* **Vacation:** 607 (3.04%)[cite: 19]

### 6. Top Loan Grades / Subgrades
* **C3:** 1,514[cite: 19]
* **C4:** 1,463[cite: 19]
* **C2:** 1,436[cite: 19]
* **C5:** 1,422[cite: 19]
* **C1:** 1,410[cite: 19]