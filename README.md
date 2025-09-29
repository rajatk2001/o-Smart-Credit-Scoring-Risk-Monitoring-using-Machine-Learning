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

## Phase 2: Data Preprocessing & EDA

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

```
## Phase 3: Model Evaluation
```python
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

# Predicted PD using logistic regression
y_pred = logit_model.predict(X_test_sm)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC:", roc_auc)

# KS statistic
ks_stat = ks_2samp(y_pred[y_test==1], y_pred[y_test==0]).statistic
print("KS Statistic:", ks_stat)
```

##Visualization
<img width="571" height="455" alt="PD Distribution" src="https://github.com/user-attachments/assets/93f4b718-cf44-43c0-88b4-892b2f294e34" />


<img width="567" height="455" alt="ROC Curve" src="https://github.com/user-attachments/assets/e9a75d03-0fdb-4de4-bbca-858b61d3dca0" />





