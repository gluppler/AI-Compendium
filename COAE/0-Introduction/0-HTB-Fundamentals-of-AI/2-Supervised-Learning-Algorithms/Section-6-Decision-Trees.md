---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 6 - Decision Trees"]
lead: Decision trees split data by feature thresholds using information gain or Gini impurity to build interpretable classification/regression models.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 6."
---

![[decision_tree.png]]

Decision trees are supervised learning models that make predictions by routing data through a series of binary feature-based decisions. Each internal node tests a feature value; each branch represents an outcome of that test; each leaf node holds a final prediction. The same structure handles both classification and regression tasks, and the resulting model is human-readable — you can trace every prediction back through the decision rules.

A decision tree has three component types:

- Root node: the top of the tree, containing the full dataset, where the first split is made.
- Internal nodes: intermediate split points, each applying a decision rule to a feature.
- Leaf nodes: terminal nodes that output the final prediction — a class label or a continuous value.

## Building a Decision Tree

At each node, the algorithm selects the feature and threshold that best separates the data. "Best" is measured by how much the split reduces impurity or uncertainty in the resulting subsets. Three common criteria:

### Gini Impurity

Gini impurity measures the probability of misclassifying a randomly chosen element if it were labeled according to the class distribution of the set. A perfectly pure node (all one class) has Gini impurity of 0.

```python
Gini(S) = 1 - Σ (pi)^2
```

Where `pi` is the proportion of class `i` in set `S`. Example with 30 class A and 20 class B instances:

```python
pA = 30 / 50 = 0.6
pB = 20 / 50 = 0.4
Gini(S) = 1 - (0.6^2 + 0.4^2)
        = 1 - (0.36 + 0.16)
        = 0.48
```

### Entropy

Entropy measures the disorder or uncertainty in a set. A pure set has entropy of 0; a perfectly mixed binary set has entropy of 1.

```python
Entropy(S) = - Σ pi * log2(pi)
```

Using the same 30/20 split:

```python
pA = 0.6, pB = 0.4
Entropy(S) = - (0.6 * log2(0.6) + 0.4 * log2(0.4))
           = - (0.6 * (-0.737) + 0.4 * (-1.322))
           = 0.971
```

### Information Gain

Information gain quantifies how much a feature reduces entropy. The algorithm selects the feature with the highest information gain at each split.

```python
Information Gain(S, A) = Entropy(S) - Σ ((|Sv| / |S|) * Entropy(Sv))
```

Where `Sv` is the subset of `S` for each value `v` of feature `A`. Example with 50 instances split by feature `F` (values 1 and 2):

- `F = 1`: 30 instances, 20 class A, 10 class B
- `F = 2`: 20 instances, 10 class A, 10 class B

```python
Entropy(S) = 0.971

# Entropy of subsets
Entropy(S1) = - (0.667 * log2(0.667) + 0.333 * log2(0.333)) = 0.918
Entropy(S2) = - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0

# Weighted average
Weighted Entropy = (30/50) * 0.918 + (20/50) * 1.0 = 0.951

# Information gain
Information Gain(S, F) = 0.971 - 0.951 = 0.020
```

### Building the Tree

The tree is built recursively: pick the best-splitting feature, partition the data, then repeat on each partition. Growth stops when any of the following is met:

- Maximum depth is reached.
- A node contains fewer than the minimum required data points.
- A node is already pure — all instances belong to one class.

Stopping criteria prevent the tree from growing deep enough to overfit.

## Playing Tennis
![[decision_tree_tennis.png]]

A concrete example: predicting whether to play tennis based on weather conditions. The training data includes features — Outlook, Temperature, Humidity, Wind — and the target label Play Tennis: Yes or No.

|PlayTennis|Outlook_Overcast|Outlook_Rainy|Outlook_Sunny|Temperature_Cool|Temperature_Hot|Temperature_Mild|Humidity_High|Humidity_Normal|Wind_Strong|Wind_Weak|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|No|False|True|False|True|False|False|False|True|False|True|
|Yes|False|False|True|False|True|False|False|True|False|True|
|No|False|True|False|True|False|False|True|False|True|False|
|No|False|True|False|False|True|False|True|False|False|True|
|Yes|False|False|True|False|False|True|False|True|False|True|
|Yes|False|False|True|False|True|False|False|True|False|True|
|No|False|True|False|False|True|False|True|False|True|False|
|Yes|True|False|False|True|False|False|True|False|False|True|
|No|False|True|False|False|True|False|False|True|True|False|
|No|False|True|False|False|True|False|True|False|True|False|

The algorithm computes information gain for each feature and selects the one with the highest value as the root. Suppose Outlook provides the most information. The root node splits into three branches: Sunny, Overcast, Rainy. The algorithm then recurses on each subset — for the Sunny subset, Humidity might give the next best split. This continues until stopping criteria are met, producing a tree whose leaf nodes output "Yes" or "No."

## Data Assumptions

Decision trees impose minimal requirements on the data:

- No linearity assumption: the model can capture non-linear relationships through sequential splits.
- No normality assumption: feature distributions do not need to be normal.
- Robustness to outliers: splits are based on threshold comparisons, not distance calculations, so extreme values affect at most one split.

This flexibility makes decision trees applicable to a wide variety of datasets without preprocessing for distribution shape or scale.

---

## Summary

- Decision trees route data through binary feature-based splits; each internal node tests a feature value and each leaf outputs a prediction.
- Splits are chosen by maximizing information gain (entropy reduction) or minimizing Gini impurity — both measure class purity in the resulting subsets.
- The tree is built recursively, stopping when maximum depth is reached, a node has too few samples, or a node is already pure.
- Decision trees can handle both classification and regression tasks and require no assumptions about feature distributions or data linearity.
- The tennis example demonstrates the full algorithm: compute information gain per feature, select the best, split, and recurse on each subset.
- Trees are highly interpretable — any prediction can be traced back through the sequence of decision rules.

---

## Best Practices

- Set maximum depth and minimum samples per leaf as stopping criteria to prevent overfitting — unconstrained trees memorize training data.
- Prefer information gain or Gini impurity for split selection; both are effective, but Gini is computationally faster.
- Decision trees do not require feature scaling or normalization — splits are threshold comparisons, not distance calculations.
- Inspect feature importances after training to identify the most predictive variables and detect potential data leakage.
- For production use, consider ensemble methods (random forests, gradient boosting) which aggregate multiple trees and reduce variance significantly.
- Visualize the trained tree to verify that the learned decision rules make domain-intuitive sense before deploying.

---

## Quiz

**Q1:** What is Gini impurity and what value indicates a perfectly pure node?
> Gini impurity is `1 - Σ(pi)²`, measuring the probability of misclassifying a randomly chosen element under the node's class distribution. A perfectly pure node (all one class) has Gini impurity of 0.

**Q2:** How does information gain guide the feature selection at each node?
> Information gain quantifies how much a feature reduces entropy in the resulting subsets. The algorithm selects the feature with the highest information gain at each split, maximizing the reduction in disorder.

**Q3:** What are the stopping criteria for tree growth?
> The tree stops growing when maximum depth is reached, a node has fewer samples than the minimum threshold, or a node is already pure (all instances belong to one class).

**Q4:** Why do decision trees not require feature normalization?
> Splits are based on threshold comparisons of individual feature values, not on distance metrics. Scale differences between features do not affect which threshold produces the best split.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-7-Naive-Bayes]] — both use entropy/probability for splitting
- see:: [[Section-8-Support-Vector-Machines]] — SVMs offer an alternative boundary approach

**Terms**
- Gini impurity, information gain, entropy, pruning, leaf node, root node, splitting criterion, tree depth
