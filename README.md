


Developed an endto-end machine learning framework to predict borrower default risk using logistic
regression and feature engineering. Applied EDA and WOE binning for interpretability, and
validated models with AUC, Gini, KS, and PSI to ensure robust performance over time





Optimization terminated successfully
         Current function value: 0.337137
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:         default_status   No. Observations:                 3500
Model:                          Logit   Df Residuals:                     3487
Method:                           MLE   Df Model:                           12
Date:                Mon, 29 Sep 2025   Pseudo R-squ.:                  0.4670
Time:                        10:50:31   Log-Likelihood:                -1180.0
converged:                       True   LL-Null:                       -2213.9
Covariance Type:            nonrobust   LLR p-value:                     0.000
=========================================================================================
                            coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------------
const                    14.5712      4.071      3.580      0.000       6.593      22.550
credit_score             -0.0180      0.007     -2.502      0.012      -0.032      -0.004
annual_income          2.977e-06   1.54e-06      1.928      0.054   -4.98e-08       6e-06
employment_length         0.0021      0.005      0.446      0.656      -0.007       0.011
dti                       0.0514      0.005      9.415      0.000       0.041       0.062
loan_amount           -6.583e-06   5.18e-06     -1.272      0.204   -1.67e-05    3.56e-06
term                      2.4001     15.083      0.159      0.874     -27.161      31.962
interest_rate            -0.0058      0.009     -0.640      0.522      -0.024       0.012
loan_purpose              0.4063      0.438      0.927      0.354      -0.453       1.266
home_ownership           -0.5615      3.474     -0.162      0.872      -7.370       6.247
income_to_loan_ratio     -0.0135      0.011     -1.288      0.198      -0.034       0.007
employment_length_cat     0.3612      0.766      0.471      0.637      -1.141       1.863
credit_score_decile      -0.4424      0.176     -2.515      0.012      -0.787      -0.098
=========================================================================================


| Feature | Coefficient | Std. Error | z-value | P>|z| | 95% CI Lower | 95% CI Upper | Interpretation |
|---------------------------|------------|------------|---------|------|--------------|--------------|----------------|
| const | 7.854 | 2.304 | 3.409 | 0.001 | 3.338 | 12.369 | Intercept |
| credit_score | -0.0093 | 0.004 | -2.294 | 0.022 | -0.017 | -0.001 | Higher score reduces default risk |
| annual_income | 1.614e-06 | 8.77e-07 | 1.840 | 0.066 | -1.05e-07 | 3.33e-06 | Slight positive effect on PD |
| employment_length | 0.0009 | 0.003 | 0.325 | 0.746 | -0.004 | 0.006 | Not significant |
| dti | 0.0290 | 0.003 | 9.480 | 0.000 | 0.023 | 0.035 | Higher DTI increases PD |
| loan_amount | -3.55e-06 | 2.95e-06 | -1.202 | 0.229 | -9.34e-06 | 2.24e-06 | Not significant |
| term | 1.118 | 8.568 | 0.130 | 0.896 | -15.675 | 17.911 | Not significant |
| interest_rate | -0.0032 | 0.005 | -0.625 | 0.532 | -0.013 | 0.007 | Not significant |
| loan_purpose | 0.2002 | 0.248 | 0.808 | 0.419 | -0.286 | 0.686 | Not significant |
| home_ownership | -0.5346 | 1.976 | -0.271 | 0.787 | -4.407 | 3.338 | Not significant |
| income_to_loan_ratio | -0.0076 | 0.006 | -1.253 | 0.210 | -0.019 | 0.004 | Slight negative effect on PD |
| employment_length_cat | 0.2453 | 0.437 | 0.561 | 0.575 | -0.612 | 1.102 | Not significant |
| credit_score_decile | -0.2796 | 0.100 | -2.799 | 0.005 | -0.475 | -0.084 | Higher decile reduces PD |

<img width="571" height="455" alt="PD Distribution" src="https://github.com/user-attachments/assets/2e708de3-e195-42e7-b66a-501663d33371" />


<img width="567" height="455" alt="ROC Curve" src="https://github.com/user-attachments/assets/b303970a-e3b6-4685-a413-480d6fa5f1dd" />
