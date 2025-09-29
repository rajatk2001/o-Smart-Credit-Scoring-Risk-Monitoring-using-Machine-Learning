


Developed an endto-end machine learning framework to predict borrower default risk using logistic
regression and feature engineering. Applied EDA and WOE binning for interpretability, and
validated models with AUC, Gini, KS, and PSI to ensure robust performance over time




### Probit Regression Model Training

```python
# Add intercept (constant term) for regression models
X_train_sm = sm.add_constant(X_train)

# Train Probit regression model
probit_model = sm.Probit(y_train, X_train_sm).fit()

# Display model summary (coefficients, p-values, etc.)
print(probit_model.summary())

<img width="571" height="455" alt="PD Distribution" src="https://github.com/user-attachments/assets/2e708de3-e195-42e7-b66a-501663d33371" />


<img width="567" height="455" alt="ROC Curve" src="https://github.com/user-attachments/assets/b303970a-e3b6-4685-a413-480d6fa5f1dd" />
