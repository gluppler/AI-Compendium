---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 11 - Principal Component Analysis"]
lead: PCA reduces dimensionality by projecting data onto principal components — eigenvectors that capture maximum variance.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 11."
---

![[pca.png]]

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional representation while preserving as much variance as possible. It works by finding new axes — the principal components — that are linear combinations of the original features and oriented along the directions of maximum data spread. PCA is used for feature extraction, noise reduction, and visualization of high-dimensional data.

![[pca_facial_features.png]]

A facial recognition database illustrates this well. Each image might have thousands of pixel features, but PCA can identify the handful of components that account for most of the variation across faces — differences in eye shape, nose width, face structure. Projecting images onto these components compresses the data while keeping what matters for comparison.

Three concepts underpin PCA:

- `Variance`: The spread of data points around the mean. Principal components are chosen to maximize variance captured.
- `Covariance`: The relationship between two features. The covariance matrix captures how features vary together across the dataset.
- Eigenvectors and Eigenvalues: Eigenvectors define the directions of the principal components; eigenvalues quantify how much variance each component explains.

The PCA algorithm:

1. `Standardize the data`: Subtract the mean and divide by the standard deviation for each feature, so that no feature dominates due to scale.
2. Calculate the covariance matrix: Compute pairwise covariances between all features.
3. Compute eigenvectors and eigenvalues: Solve the eigenvalue decomposition of the covariance matrix.
4. `Sort eigenvectors`: Order eigenvectors by their eigenvalues in descending order — the first captures the most variance.
5. Select principal components: Keep the top `k` eigenvectors that collectively explain the desired amount of variance.
6. `Transform the data`: Project the original data onto the selected components to get the reduced representation.

## Eigenvalues and Eigenvectors

An `eigenvector` of a matrix `A` is a non-zero vector `v` that satisfies:

$$A \mathbf{v} = \lambda \mathbf{v}$$

The scalar `λ` is the corresponding `eigenvalue`. When `A` transforms `v`, the result points in the same direction as `v` but is scaled by `λ`. The vector's direction is preserved — only its magnitude changes.

![[eigenvector_rubber_band.png]]

**Example**: Given the transformation matrix:

$$A = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$$

Applying `A` to the vector `v = [1, 0]`:

$$A \mathbf{v} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 0 \end{bmatrix}$$

The result `[2, 0]` points in the same direction as `[1, 0]` but is scaled by 2. So `v = [1, 0]` is an eigenvector with eigenvalue `λ = 2`.

### The Eigenvalue Equation in Principal Component Analysis (PCA)

In PCA, the eigenvalue equation is applied to the covariance matrix `C`:

$$C \mathbf{v} = \lambda \mathbf{v}$$

- `C` is the covariance matrix of the standardized data.
- `v` is the eigenvector — defines a principal component direction.
- `λ` is the eigenvalue — quantifies how much variance that component captures. Larger eigenvalue = more variance explained.

### Solving the Eigenvalue Equation

Two main approaches:

- Eigenvalue Decomposition: Directly computes eigenvectors and eigenvalues from the covariance matrix.
- Singular Value Decomposition (SVD): Decomposes the data matrix directly. More numerically stable and the method used in most implementations (e.g., `sklearn`'s `PCA`).

### Selecting Principal Components

After sorting eigenvectors by descending eigenvalue, select the top `k` to form the projection matrix `V`. The transformation:

$$Y = X \cdot V$$

Projects the original data matrix `X` (shape: samples × features) onto the `k`-dimensional space defined by the selected eigenvectors, yielding `Y` (shape: samples × k).

## Choosing the Number of Components

Plot the cumulative explained variance ratio against the number of components. Each component's ratio is its eigenvalue divided by the sum of all eigenvalues. A common threshold is 95% — retain enough components to explain 95% of total variance. The plot reveals whether a small number of components dominates (concentrated variance) or whether variance is spread across many components (harder to compress).

## Data Assumptions

- `Linearity`: PCA finds linear combinations of features. Non-linear structure requires methods like kernel PCA or autoencoders.
- `Correlation`: PCA is most effective when features are correlated — uncorrelated features already represent independent directions, so PCA adds little.
- `Scale sensitivity`: Features with larger numeric ranges dominate the covariance matrix. Standardize before applying PCA.

---

## Summary

- PCA reduces dimensionality by projecting data onto principal components — eigenvectors of the covariance matrix — that capture maximum variance.
- The six-step algorithm: standardize → compute covariance matrix → eigendecompose → sort by eigenvalue → select top k components → project data.
- Eigenvalues quantify how much variance each principal component explains; eigenvectors define the direction of each component in the original feature space.
- Singular Value Decomposition (SVD) is the numerically stable method used in most implementations (e.g., sklearn) instead of direct eigendecomposition.
- The cumulative explained variance plot guides the choice of k components; a common threshold is retaining enough to explain 95% of total variance.
- PCA assumes linear relationships and is most effective when features are correlated; it also requires standardization to prevent high-scale features from dominating.

---

## Best Practices

- Always standardize features before PCA — features with large numeric ranges will dominate the covariance matrix and distort the principal components.
- Use the cumulative explained variance plot to choose k; do not use a fixed number — the right k depends on the data's intrinsic dimensionality.
- Apply PCA before clustering or visualization to reduce noise and computational cost, not as a replacement for good feature engineering.
- Verify that features are correlated before applying PCA — uncorrelated features already span independent directions and PCA adds no compression benefit.
- For non-linear structure, use kernel PCA or autoencoders; standard PCA only captures linear combinations of features.
- Retain the PCA transform fitted on training data and apply the same transform to test data — never refit PCA on test data.

---

## Quiz

**Q1:** What do eigenvalues and eigenvectors of the covariance matrix represent in PCA?
> Eigenvectors define the directions of the principal components (the new axes). Eigenvalues quantify how much variance each component captures — a larger eigenvalue means that component explains more of the total data spread.

**Q2:** Why must data be standardized before PCA?
> PCA identifies directions of maximum variance. Without standardization, features with large numeric ranges dominate the covariance matrix and produce components that reflect scale differences rather than true structure.

**Q3:** How is the number of principal components typically chosen?
> Plot cumulative explained variance ratio vs. number of components and choose k such that a target threshold (commonly 95%) is reached. This balances dimensionality reduction against information retention.

**Q4:** What is the difference between eigenvalue decomposition and SVD in the context of PCA, and which is preferred?
> Eigenvalue decomposition operates on the square covariance matrix. SVD decomposes the data matrix directly and is more numerically stable — it avoids computing the covariance matrix explicitly and handles cases where n < d. SVD is the preferred and default method in most libraries.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-2-Mathematics-Refresher-for-AI]] — eigenvalues and eigenvectors are the mathematical basis of PCA
- see:: [[Section-10-K-Means-Clustering]] — PCA is often applied before clustering to reduce noise

**Terms**
- principal components, eigenvectors, eigenvalues, variance explained, covariance matrix, dimensionality reduction, feature extraction
