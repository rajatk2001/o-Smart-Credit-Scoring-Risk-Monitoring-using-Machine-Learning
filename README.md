# Probability of Default (PD) Scorecard Project

## Project Overview
Developed an **end-to-end machine learning framework** to predict borrower default risk using logistic regression and feature engineering.  
Applied **EDA and WOE binning** for interpretability, and validated models with **AUC, Gini, KS, and PSI** to ensure robust performance over time.

---

## Dataset
- **Source:** Synthetic Data  
- **Features:**
  - `credit_score`, `annual_income`, `employment_length`, `dti`, `loan_amount`, `term`, `interest_rate`
  - `loan_purpose`, `home_ownership`
  - Derived features: `credit_utilization_ratio`, `income_to_loan_ratio`
- **Target:** `default_status` (1 = defaulted, 0 = fully paid)

---

## Phase 2: Data Preprocessing

```python
# Convert percentage columns to float
def percent_to_float(x):
    if isinstance(x, str):
        return float(x.strip('%'))
    else:
        return float(x)

df['dti'] = df['dti'].apply(percent_to_float)
df['interest_rate'] = df['interest_rate'].apply(percent_to_float)

# Clean annual_income
df['annual_income'] = df['annual_income'].replace(',','', regex=True).astype(float)

# Ensure numeric columns are float
numeric_cols = ['credit_score', 'annual_income', 'employment_length', 'dti', 'loan_amount', 'interest_rate']
df[numeric_cols] = df[numeric_cols].astype(float)

<img width="567" height="455" alt="ROC Curve" src="https://github.com/user-attachments/assets/2256a8b7-0ee0-467b-86f0-383b7ddbffba" />




