---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 8 - Support Vector Machines"]
lead: SVMs find the maximum-margin hyperplane separating classes, using the kernel trick to handle non-linear boundaries.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 8."
---

![[svm.png]]

Support Vector Machines (SVMs) are `supervised learning` algorithms used for both `classification` and `regression`. They perform well on high-dimensional data and can model complex non-linear decision boundaries via the kernel trick. The central idea is to find the `hyperplane` that separates classes with the maximum possible margin.

## Maximizing the Margin

The `margin` is the distance between the decision boundary (hyperplane) and the nearest training points from each class. Those nearest points are the `support vectors` — they alone determine the position and orientation of the hyperplane. All other training points are irrelevant once the support vectors are identified.

A larger margin reduces sensitivity to noise in the training data and improves generalization to unseen examples.

## Linear SVM

A `linear SVM` applies when classes are linearly separable — a single hyperplane can correctly separate them.

### Finding the Optimal Hyperplane

![[svm_optimal_hyperplane.png]]

The hyperplane is defined by:

$$\mathbf{w} \cdot \mathbf{x} + b = 0$$

Where `w` is the weight vector (perpendicular to the hyperplane) and `b` is the bias. Training learns the values of `w` and `b` that maximize the margin while classifying all points correctly.

Consider classifying emails as spam or not spam using the word frequencies of "free" and "money" as features. Plotting each email in this two-dimensional feature space, the linear SVM finds the line (hyperplane in 2D) that sits furthest from both the nearest spam and nearest non-spam emails.

## Non-Linear SVM

![[svm_non_linear.png]]

When classes cannot be separated by a straight line or flat hyperplane, `non-linear SVMs` map the data to a higher-dimensional space where linear separation becomes possible.

### Kernel Trick

The `kernel trick` achieves this mapping implicitly — the kernel function computes dot products in the higher-dimensional space without explicitly constructing the transformed feature vectors. This is computationally efficient and avoids the need to define the mapping explicitly.

### Kernel Functions

- `Polynomial Kernel`: Introduces polynomial feature interactions (e.g., $x^2$, $x^3$), adding curvature to the decision boundary.
- Radial Basis Function (RBF) Kernel: Maps data using a Gaussian function. Highly flexible and the most commonly used default; captures complex non-linear patterns.
- `Sigmoid Kernel`: Applies a sigmoid-shaped transformation, similar to logistic regression's activation function.

The right kernel depends on the data structure and the complexity needed. RBF is a reasonable starting point for most problems.

### Image Classification

Images of cats and dogs differ along non-linear feature combinations — fur texture, ear shape, facial geometry. A `non-linear SVM` with an RBF kernel can learn these boundaries effectively, making SVMs a historically strong baseline for image classification before deep learning took over.

## The SVM Function

The optimization problem for a linear SVM is:

$$\text{Minimize: } \frac{1}{2} \|\mathbf{w}\|^2$$
$$\text{Subject to: } y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \text{ for all } i$$

Where `y_i ∈ {-1, +1}` is the class label for training point `x_i`. Minimizing `‖w‖²` is equivalent to maximizing the margin, while the constraint ensures all training points are correctly classified with a margin of at least 1.

## Data Assumptions

- No Distributional Assumptions: SVMs make no assumptions about the underlying data distribution.
- Handles High Dimensionality: Effective when the number of features exceeds the number of samples — common in text classification and genomics.
- `Robust to Outliers`: The margin maximization focuses on support vectors rather than fitting every point, so individual outliers have limited influence.

---

## Summary

- SVMs find the maximum-margin hyperplane separating classes — a larger margin reduces sensitivity to noise and improves generalization.
- Support vectors are the training points nearest the decision boundary; they alone determine the hyperplane's position and orientation.
- Linear SVMs work when classes are linearly separable; non-linear SVMs use the kernel trick to map data to a higher-dimensional space where linear separation becomes possible.
- The kernel trick computes dot products in the transformed space without explicitly constructing it — making complex boundaries computationally efficient.
- Common kernels: Polynomial (adds curvature), RBF (flexible, default choice), Sigmoid (logistic-like boundary).
- The SVM optimization minimizes `‖w‖²/2` subject to the margin constraint `y_i(w·x_i + b) ≥ 1`, ensuring all points are correctly classified with maximum separation.

---

## Best Practices

- Start with the RBF kernel as a default for non-linear problems; it handles most complex patterns and has only one hyperparameter (gamma) to tune alongside C.
- Standardize features before training SVMs — the margin calculation uses Euclidean distance, so differing feature scales directly distort the margin.
- Tune the C parameter (soft margin penalty) carefully: high C fits training data closely but risks overfitting; low C allows more margin violations for better generalization.
- Use SVMs as a strong baseline for high-dimensional sparse data (e.g., text classification) where they historically outperform many other methods.
- For very large datasets, consider approximate SVM methods or switch to gradient boosting / neural networks, as standard SVM training scales poorly with sample count.

---

## Quiz

**Q1:** What is the margin in an SVM, and why is maximizing it beneficial?
> The margin is the distance between the decision hyperplane and the nearest training points (support vectors) from each class. Maximizing it reduces sensitivity to noise and improves generalization to unseen data.

**Q2:** What are support vectors and why are other training points irrelevant once they are identified?
> Support vectors are the training points closest to the decision boundary. The hyperplane's position is determined entirely by these points; all other training points lie outside the margin and do not affect the solution.

**Q3:** What problem does the kernel trick solve in non-linear SVMs?
> Classes may not be linearly separable in the original feature space. The kernel trick implicitly maps data to a higher-dimensional space where linear separation is possible, without explicitly computing the transformation — keeping computation tractable.

**Q4:** Write and explain the SVM optimization objective.
> Minimize `‖w‖²/2` subject to `y_i(w·x_i + b) ≥ 1` for all i. Minimizing `‖w‖²` maximizes the margin width (2/‖w‖), while the constraint ensures every training point is correctly classified with at least unit margin.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-2-Mathematics-Refresher-for-AI]] — norms and vector operations underpin margin calculations
- see:: [[Section-6-Decision-Trees]] — decision trees are an alternative for non-linear boundaries

**Terms**
- hyperplane, support vectors, margin, kernel trick, soft margin, C parameter, RBF kernel, dual problem
