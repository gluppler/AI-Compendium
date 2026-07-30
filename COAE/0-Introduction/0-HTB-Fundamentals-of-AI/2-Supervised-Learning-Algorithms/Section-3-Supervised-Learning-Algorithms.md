---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 3 - Supervised Learning Algorithms"]
lead: Supervised learning trains on labeled data to predict outputs — covers classification, regression, and core concepts like overfitting and cross-validation.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 3."
---

Supervised learning trains a model on labeled data — examples where the correct output is already known — so the model can predict outputs for new, unseen inputs. The algorithm learns a mapping from input features to output labels by minimizing the difference between its predictions and the ground truth.

Supervised problems split into two types:

1. Classification: predict a discrete category (e.g., spam vs. not spam, cat vs. dog vs. bird).
2. Regression: predict a continuous value (e.g., house price, tomorrow's temperature).

## How Supervised Learning Works

Training data is a labeled dataset of input-output pairs. Features are the measurable input variables (e.g., house size, number of bedrooms, location). Labels are the target outputs — the "correct answers" the model learns to reproduce (e.g., the actual sale price). The model is a mathematical function that maps features to a predicted label, with parameters that are adjusted during training to minimize prediction error. Once trained, the model performs inference on new data: given features, it outputs a prediction.

Prediction and inference are related but distinct. Prediction generates actionable outputs — classifying an email or forecasting a price. Inference is broader: it also encompasses estimating model parameters, identifying which features are most important, and understanding how inputs affect outputs.

Evaluation measures how well the model performs on held-out data. Common metrics:

- Accuracy: fraction of correct predictions.
- Precision: fraction of positive predictions that are actually positive.
- Recall: fraction of actual positives the model correctly identified.
- F1-score: harmonic mean of precision and recall.

Generalization is the goal — a model that performs well on data it has never seen, not just on the training set.

## Core Concepts in Supervised Learning

### Overfitting

Overfitting occurs when a model learns the training data too precisely — capturing noise and outliers rather than the underlying pattern. The result is high training accuracy but poor performance on new data. An overfitted model has memorized rather than learned.

### Underfitting

Underfitting occurs when a model is too simple to capture the structure in the data. Performance is poor on both training and test sets. It typically results from an insufficiently expressive model or too little training.

### Cross-Validation

Cross-validation estimates how well a model will generalize before committing to a final evaluation on held-out test data. The dataset is split into `k` folds; the model trains on `k-1` folds and validates on the remaining one. This repeats `k` times with each fold serving as the validation set once. The average validation score is a more reliable generalization estimate than a single train/test split, and it reduces the risk of overfitting to a particular data split.

### Regularization

Regularization reduces overfitting by adding a penalty to the loss function that discourages large or complex parameter values. The two main forms:

- L1 regularization: penalizes the sum of absolute coefficient values. Tends to drive some coefficients to exactly zero, producing sparse models.
- L2 regularization: penalizes the sum of squared coefficient values. Shrinks coefficients toward zero without eliminating them, producing smoother models.

Both force the model to prioritize simpler patterns that are more likely to generalize.

---

## Summary

- Supervised learning trains on labeled input-output pairs so the model can predict outputs for unseen inputs by minimizing prediction error.
- The two problem types are classification (discrete category output) and regression (continuous value output).
- Key evaluation metrics are accuracy, precision, recall, and F1-score — each captures a different aspect of model performance on held-out data.
- Overfitting occurs when a model memorizes training noise; underfitting occurs when the model is too simple to capture the signal.
- Cross-validation estimates generalization by rotating through k data folds, providing a more reliable estimate than a single train/test split.
- L1 regularization drives sparse coefficients (feature selection); L2 regularization shrinks all coefficients smoothly toward zero.

---

## Best Practices

- Always evaluate on a held-out test set that was never used during training or hyperparameter selection to get an unbiased generalization estimate.
- Use k-fold cross-validation (typically k=5 or 10) rather than a single split when the dataset is small.
- Choose precision, recall, or F1 over accuracy when class imbalance is present — accuracy can be misleadingly high on imbalanced data.
- Apply regularization (L1 or L2) as a default when training on high-dimensional data or small datasets to combat overfitting.
- Diagnose overfitting and underfitting by comparing training vs. validation loss — a large gap signals overfitting; both high indicates underfitting.
- Select L1 regularization when feature selection is desired; use L2 when all features are expected to contribute and smooth shrinkage is preferred.

---

## Quiz

**Q1:** What is the difference between classification and regression in supervised learning?
> Classification predicts a discrete category (e.g., spam or not spam); regression predicts a continuous numeric value (e.g., house price or temperature).

**Q2:** What is overfitting and how can it be detected?
> Overfitting is when a model learns training data noise rather than generalizable patterns. It is detected by a gap between high training accuracy and substantially lower validation/test accuracy.

**Q3:** How does k-fold cross-validation work and why is it preferred over a single train/test split?
> The dataset is divided into k folds; the model trains on k-1 folds and validates on the remaining one, repeating k times. The averaged score is more reliable because it reduces variance from any particular data split.

**Q4:** What is the key behavioral difference between L1 and L2 regularization?
> L1 penalizes the sum of absolute coefficient values, driving some coefficients to exactly zero (sparse model). L2 penalizes the sum of squared values, shrinking all coefficients toward zero without eliminating them.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-4-Linear-Regression]] — regression example
- see:: [[Section-5-Logistic-Regression]] — classification example
- see:: [[Section-6-Decision-Trees]] — tree-based classification/regression
- see:: [[Section-7-Naive-Bayes]] — probabilistic classification
- see:: [[Section-8-Support-Vector-Machines]] — margin-based classification

**Terms**
- training data, features, labels, overfitting, underfitting, cross-validation, regularization, generalization, classification, regression, inference
