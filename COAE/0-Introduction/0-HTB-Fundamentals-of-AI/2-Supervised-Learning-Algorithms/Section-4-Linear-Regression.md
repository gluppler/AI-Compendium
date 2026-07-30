---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 4 - Linear Regression"]
lead: Linear regression finds the best-fit line minimizing squared residuals to predict continuous target variables.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 4."
---

![[linear_regression.png]]

Linear regression is a supervised learning algorithm that models a linear relationship between one or more predictor variables and a continuous target variable. The algorithm finds coefficient values that define the best-fitting line (or hyperplane) through the data by minimizing the sum of squared differences between predictions and actual values.

## What is Regression?

Regression is the supervised learning task of predicting a continuous output — a number that can take any value in a range, rather than a discrete category. Examples:

- Predicting house price from size, location, and age.
- Forecasting daily temperature from historical weather data.
- Estimating site traffic from marketing spend and seasonality.

This distinguishes regression from classification, where the output is a categorical label. Linear regression is regression under the specific assumption that the relationship between predictors and the target is linear.

## Simple Linear Regression

Simple linear regression uses one predictor variable to predict one target variable:

```python
y = mx + c
```

Where:

- `y` is the predicted target variable
- `x` is the predictor variable
- `m` is the slope — the change in y per unit change in x
- `c` is the y-intercept — the predicted value of `y` when `x` is 0

Training finds the `m` and `c` values that minimize the error between predicted and actual `y` values.

## Multiple Linear Regression

When multiple predictors are used:

```python
y = b0 + b1*x1 + b2*x2 + ... + bn*xn
```

Where:

- `y` is the predicted target variable
- `x1, x2, ..., xn` are the predictor variables
- `b0` is the y-intercept
- `b1, b2, ..., bn` are the coefficients, one per predictor

## Ordinary Least Squares
![[ols.png]]

Ordinary Least Squares (OLS) is the standard method for estimating linear regression coefficients. It minimizes the Residual Sum of Squares (RSS) — the sum of squared differences between actual and predicted values.

The process:

1. Compute residuals: for each data point, `residual = actual_y - predicted_y`.
2. Square each residual: this makes all values positive and penalizes larger errors more heavily.
3. Sum the squared residuals to get the RSS.
4. Adjust coefficients to minimize the RSS.

The result is the line that minimizes the total squared distance between the data points and the line.

## Assumptions of Linear Regression

Linear regression relies on four assumptions about the data:

- Linearity: the relationship between predictors and the target is linear.
- Independence: observations do not influence each other.
- Homoscedasticity: the variance of residuals is constant across all predictor values — the spread of errors does not change with the magnitude of predictions.
- Normality: residuals are normally distributed — required for valid statistical inference on coefficients.

Violating these assumptions does not always break predictions, but it can make coefficient estimates unreliable and invalidate hypothesis tests on the model.

---

## Summary

- Linear regression models a linear relationship between one or more predictor variables and a continuous target variable.
- Simple linear regression uses one predictor (`y = mx + c`); multiple linear regression uses several predictors (`y = b0 + b1*x1 + ... + bn*xn`).
- Ordinary Least Squares (OLS) estimates coefficients by minimizing the Residual Sum of Squares — the sum of squared differences between actual and predicted values.
- The four assumptions of linear regression are linearity, independence, homoscedasticity, and normality of residuals.
- Violating assumptions does not always break predictions but can make coefficient estimates unreliable and invalidate hypothesis tests.
- Linear regression is the foundation for logistic regression and many other supervised learning algorithms.

---

## Best Practices

- Always check the four assumptions (linearity, independence, homoscedasticity, normality) before trusting coefficient estimates for inference.
- Plot residuals vs. fitted values to check homoscedasticity — a fan-shaped pattern indicates violation and may require transformation or robust regression.
- Standardize or scale predictors before comparing coefficient magnitudes, since raw coefficients are scale-dependent.
- Check for multicollinearity among predictors using VIF (Variance Inflation Factor) — high multicollinearity inflates standard errors and destabilizes estimates.
- Use regression for continuous target prediction; if the target is categorical, switch to logistic regression or a classification algorithm.

---

## Quiz

**Q1:** What is the residual in linear regression, and what does OLS minimize?
> A residual is the difference between the actual and predicted value for a data point. OLS minimizes the Residual Sum of Squares — the sum of all squared residuals.

**Q2:** Why does OLS square the residuals rather than sum them directly?
> Squaring makes all residuals positive (preventing positive and negative errors from cancelling) and penalizes larger errors more heavily than smaller ones.

**Q3:** What is homoscedasticity and why does it matter?
> Homoscedasticity means the variance of residuals is constant across all predictor values. Violations make standard error estimates unreliable, invalidating confidence intervals and hypothesis tests on the coefficients.

**Q4:** What is the key difference between simple and multiple linear regression?
> Simple linear regression uses one predictor variable; multiple linear regression uses two or more, with one coefficient per predictor plus an intercept.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-5-Logistic-Regression]] — logistic extends linear for classification tasks
- see:: [[Section-2-Mathematics-Refresher-for-AI]] — OLS uses matrix operations and norms

**Terms**
- ordinary least squares, mean squared error, gradient descent, cost function, residuals, regression coefficient, R-squared
