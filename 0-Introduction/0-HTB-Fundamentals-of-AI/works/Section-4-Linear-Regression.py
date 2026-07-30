# Source: HackTheBox, Fundamentals of AI, Section 4 — Linear Regression
# Section file: ../2-Supervised-Learning-Algorithms/Section-4-Linear-Regression.md
# All fenced code blocks extracted verbatim in the order they appear in the note.

# ------------------------------------------------------------
# Simple Linear Regression formula
# ------------------------------------------------------------

# y = mx + c
# y — predicted target variable
# x — predictor variable
# m — slope (change in y per unit change in x)
# c — y-intercept (predicted y when x = 0)
y = m * x + c

# ------------------------------------------------------------
# Multiple Linear Regression formula
# ------------------------------------------------------------

# y = b0 + b1*x1 + b2*x2 + ... + bn*xn
# y          — predicted target variable
# x1...xn    — predictor variables
# b0         — y-intercept
# b1...bn    — coefficients, one per predictor
y = b0 + b1*x1 + b2*x2 + ... + bn*xn
