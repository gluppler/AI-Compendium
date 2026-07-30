---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 5 - Logistic Regression"]
lead: Logistic regression uses the sigmoid function to model binary classification despite the "regression" name.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 5."
---

![[logistic_regression_classification.png]]

Despite its name, logistic regression is a classification algorithm, not a regression one. It predicts a binary categorical outcome (0 or 1, yes or no, spam or not spam) by computing the probability that an input belongs to the positive class. When that probability exceeds a threshold, the model assigns the positive label.

## What is Classification?

Classification is supervised learning for discrete outputs. The model assigns each input to one of a fixed set of categories, unlike regression which predicts continuous values. Examples:

- Identifying fraudulent transactions (fraudulent / not fraudulent)
- Classifying images of animals (cat / dog / bird)
- Diagnosing disease presence from patient symptoms

## How Logistic Regression Works

Logistic regression runs a linear combination of the input features through a sigmoid function to produce a probability between 0 and 1. That probability is then thresholded to make a binary decision.

### What is a Sigmoid Function?
![[sigmoid.png]]

The sigmoid function maps any real-valued number to the range (0, 1). Its S-shaped curve starts near 0 for large negative inputs, passes through 0.5 at zero, and approaches 1 for large positive inputs. This smooth mapping makes it suitable for representing probabilities.

### The Sigmoid Function

$$P(x) = \frac{1}{1 + e^{-z}}$$

Where:

- `P(x)` is the predicted probability
- `e` is Euler's number (≈ 2.718)
- `z` is the linear combination of input features and learned weights: `z = m1*x1 + m2*x2 + ... + mn*xn + c`

### Spam Detection

A spam filter built with logistic regression computes `z` from features like sender address, keyword presence, and email length, then runs `z` through the sigmoid to get a spam probability. Emails above the threshold (e.g., 0.5) are classified as spam.

### Decision Boundary
![[logistic_regression.png]]

The decision boundary is the set of points where the model's predicted probability equals the threshold. In a two-feature space this is a line; with more features it becomes a hyperplane. Points on one side are classified as positive, points on the other as negative. The boundary is determined by the learned weights and the chosen threshold.

## Understanding Hyperplanes
![[hyperplane.png]]

A hyperplane is a flat subspace of dimension one less than the ambient space. In 2D it is a line; in 3D it is a plane. In higher dimensions — which is the typical case in ML — it cannot be visualized directly, but the geometry is the same: one flat surface divides the space into two regions. In logistic regression, the hyperplane is the decision boundary separating the two classes.

### Threshold Probability

The default threshold is 0.5 but it is adjustable. Raising it requires higher model confidence before assigning the positive label, reducing false positives at the cost of more false negatives. The right threshold depends on the relative cost of each error type in the application.

## Data Assumptions

Logistic regression makes fewer assumptions than linear regression, but several still apply:

- Binary outcome: the target variable must have exactly two classes.
- Linearity of log odds: the log of the odds ratio must have a linear relationship with the predictor variables. This is the internal assumption the model encodes, not a requirement on raw probabilities.
- No strong multicollinearity: highly correlated predictors make it hard to isolate each variable's effect on the outcome.
- Sufficient sample size: parameter estimates become unreliable with small datasets.

---

## Summary

- Despite its name, logistic regression is a classification algorithm that predicts the probability of binary class membership using the sigmoid function.
- The sigmoid function maps any real value to (0, 1), producing a probability; a threshold (default 0.5) converts the probability to a binary label.
- The linear combination of features `z = m1*x1 + ... + mn*xn + c` is passed through the sigmoid, and the decision boundary is where this output equals the threshold.
- A hyperplane is the decision boundary — a line in 2D, a plane in 3D, a flat subspace in higher dimensions — dividing the feature space into two class regions.
- Adjusting the threshold trades off precision vs. recall: higher threshold reduces false positives; lower threshold reduces false negatives.
- Key assumptions: binary outcome, linearity of log-odds, no strong multicollinearity, and sufficient sample size.

---

## Best Practices

- Treat the 0.5 decision threshold as a starting point, not a fixed rule — tune it based on the relative costs of false positives vs. false negatives in the application domain.
- Check for multicollinearity among predictors before fitting; highly correlated features destabilize coefficient estimates.
- Ensure sufficient sample size per class — logistic regression estimates become unreliable with fewer than ~10 events per predictor variable (EPV rule of thumb).
- Evaluate with precision, recall, and AUC-ROC rather than accuracy alone, especially when classes are imbalanced.
- Regularize (L1 or L2) when the feature space is wide relative to sample size to prevent coefficient inflation.

---

## Quiz

**Q1:** Why is logistic regression classified as a classification algorithm despite the word "regression" in its name?
> It predicts a probability using a regression-like linear combination of inputs, but the final output is a binary class label determined by thresholding that probability — making it a classifier, not a regressor.

**Q2:** What is the sigmoid function and what property makes it suitable for probability modeling?
> The sigmoid maps any real number to the open interval (0, 1) with an S-shaped curve. This bounded output can be interpreted as a probability, satisfying the requirement that probabilities lie between 0 and 1.

**Q3:** What is the decision boundary in logistic regression?
> The set of points where the model's predicted probability equals the chosen threshold. In a two-feature space it is a line; with more features it becomes a hyperplane. Points on either side are assigned to opposite classes.

**Q4:** What does raising the classification threshold from 0.5 to 0.8 do to the model's behavior?
> It requires the model to be more confident before assigning the positive label, reducing false positives but increasing false negatives — the model becomes more conservative in its positive predictions.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-4-Linear-Regression]] — logistic builds on linear regression concepts
- see:: [[Section-7-Naive-Bayes]] — both are probabilistic classifiers
- see:: [[Section-2-Mathematics-Refresher-for-AI]] — exponential and probability notation

**Terms**
- sigmoid function, log loss, decision boundary, binary classification, odds ratio, log-odds, cross-entropy
