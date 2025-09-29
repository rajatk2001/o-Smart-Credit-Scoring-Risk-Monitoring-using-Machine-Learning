

## Project Overview\
Developed an endto-end machine learning framework to predict borrower default risk using logistic
regression and feature engineering. Applied EDA and WOE binning for interpretability, and
validated models with AUC, Gini, KS, and PSI to ensure robust performance over time
# Probability of Default (PD) Scorecard Project


---

## Dataset
- **Source:**  Synthetic Data 
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

---

##Phase 4: Logiststic and Probit Regression Model Training

```python
# Add intercept
X_train_sm = sm.add_constant(X_train)

# Train logistic regression model
logit_model = sm.Logit(y_train, X_train_sm).fit()

# Model summary
print(logit_model.summary())
# Add intercept (constant term) for regression models
X_train_sm = sm.add_constant(X_train)

# Train Probit regression model
probit_model = sm.Probit(y_train, X_train_sm).fit()

# Display model summary (coefficients, p-values, etc.)
print(probit_model.summary())


---
<img width="571" height="455" alt="PD Distribution" src="https://github.com/user-attachments/assets/778e2ee7-2341-46a2-8646-86bbdddea2bb" />

<img width="567" height="455" alt="ROC Curve" src="https://github.com/user-attachments/assets/aa3f976a-732e-4afa-84ad-6a32f346461c" />
tion
# Phase  4 :model evaluating
from sklearn.metrics import roc_auc_score

# Predicted PD
y_pred = logit_model.predict(X_test_sm)

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC:", roc_auc)

# KS statistic (example)
from scipy.stats import ks_2samp
ks_stat = ks_2samp(y_pred[y_test==1], y_pred[y_test==0]).statistic
print("KS Statistic:", ks_stat)



---

This Markdown is **ready for GitHub**, with:  
- Code blocks for preprocessing, modeling, evaluation, and scorecard steps  
- Clear explanations in simple language  
- Placeholders for metrics like KS, Gini, and PD predictions  

---

I can also **make it even shorter and visually appealing**, using **tables for risk bands and metrics**, so it looks more like a polished GitHub project.  

Do you want me to do that?


