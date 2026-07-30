---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 2 - Mathematics Refresher for AI"]
lead: Reference for mathematical notation used across AI — linear algebra, probability, and statistical operators.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 2."
---

This page is a notation reference. The symbols below appear throughout the module's algorithms and formulas. You do not need to memorize all of it — return here when a symbol is unfamiliar.

## Basic Arithmetic Operations

### Multiplication (`*`)

The `*` operator denotes the product of two numbers or expressions:

```python
3 * 4 = 12
```

### Division (`/`)

The `/` operator denotes dividing one number by another:

```python
10 / 2 = 5
```

### Addition (`+`)

The `+` operator represents the sum of two or more values:

```python
5 + 3 = 8
```

### Subtraction (`-`)

The `-` operator represents the difference between two values:

```python
9 - 4 = 5
```

## Algebraic Notations

### Subscript Notation (`x_t`)

A subscript indexes a variable, typically indicating a position in a sequence or a time step. `x_t` is the value of `x` at time `t`:

```python
x_t = q(x_t | x_{t-2})
```

### Superscript Notation (`x^n`)

Superscripts denote exponents or powers:

```python
x^2 = x * x
```

### Norm (`||...||`)

The norm measures the size of a vector. The Euclidean (L2) norm is:

```python
||v|| = sqrt(v_1^2 + v_2^2 + ... + v_n^2)
```

Other norms:

```python
||v||_1 = |v_1| + |v_2| + ... + |v_n|
||v||_∞ = max(|v_1|, |v_2|, ..., |v_n|)
```

Norms appear in distance calculations, regularization, and data normalization.

### Summation Symbol (`Σ`)

The summation symbol indicates adding all terms in a sequence from index `i=1` to `n`:

```python
Σ_{i=1}^{n} a_i
```

This represents `a_1 + a_2 + ... + a_n`. Summation is used in means, variances, and loss functions.

## Logarithms and Exponentials

### Logarithm Base 2 (`log2(x)`)

The base-2 logarithm asks how many times you must multiply 2 to get `x`. It appears in information theory (entropy):

```python
log2(8) = 3
```

### Natural Logarithm (`ln(x)`)

The natural logarithm uses base `e` (Euler's number ≈ 2.718). It is the inverse of the exponential function:

```python
ln(e^2) = 2
```

Used in calculus, differential equations, and probability (log-likelihood, cross-entropy loss).

### Exponential Function (`e^x`)

`e` raised to the power of `x`. It models continuous growth and decay:

```python
e^2 ≈ 7.389
```

### Exponential Function (Base 2) (`2^x`)

2 raised to the power of `x`. Common in binary systems and information metrics:

```python
2^3 = 8
```

## Matrix and Vector Operations

### Matrix-Vector Multiplication (`A * v`)

Multiplies each row of matrix `A` by vector `v` to produce a new vector. Fundamental in linear transformations and neural network forward passes:

```python
A * v = [[1, 2], [3, 4]] * [5, 6] = [17, 39]
```

### Matrix-Matrix Multiplication (`A * B`)

Each element of the output is the dot product of a row from `A` and a column from `B`:

```python
A * B = [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
```

Used in linear transformations and between layers in deep networks.

### Transpose (`A^T`)

Swaps rows and columns of matrix `A`:

```python
A   = [[1, 2], [3, 4]]
A^T = [[1, 3], [2, 4]]
```

Used in dot products and data preparation.

### Inverse (`A^{-1}`)

The matrix that, when multiplied by `A`, yields the identity matrix:

```python
A      = [[1, 2], [3, 4]]
A^{-1} = [[-2, 1], [1.5, -0.5]]
```

Used for solving linear systems and inverting transformations.

### Determinant (`det(A)`)

A scalar derived from a square matrix. Non-zero determinant means the matrix is invertible:

```python
A      = [[1, 2], [3, 4]]
det(A) = 1 * 4 - 2 * 3 = -2
```

Used in geometric transformations and volume calculations.

### Trace (`tr(A)`)

The sum of the diagonal elements of a square matrix:

```python
A     = [[1, 2], [3, 4]]
tr(A) = 1 + 4 = 5
```

Used in matrix properties and eigenvalue calculations.

## Set Theory

### Cardinality (`|S|`)

The number of elements in set `S`:

```python
S = {1, 2, 3, 4, 5}
|S| = 5
```

### Union (`∪`)

All elements in either `A` or `B` or both:

```python
A = {1, 2, 3}, B = {3, 4, 5}
A ∪ B = {1, 2, 3, 4, 5}
```

### Intersection (`∩`)

Elements that appear in both `A` and `B`:

```python
A = {1, 2, 3}, B = {3, 4, 5}
A ∩ B = {3}
```

### Complement (`A^c`)

All elements in the universal set `U` that are not in `A`:

```python
U = {1, 2, 3, 4, 5}, A = {1, 2, 3}
A^c = {4, 5}
```

## Comparison Operators

### Greater Than or Equal To (`>=`)

```python
a >= b
```

### Less Than or Equal To (`<=`)

```python
a <= b
```

### Equality (`==`)

```python
a == b
```

### Inequality (`!=`)

```python
a != b
```

## Eigenvalues and Scalars

### Lambda (Eigenvalue) (`λ`)

An eigenvalue `λ` satisfies `A * v = λ * v` — multiplying vector `v` by matrix `A` scales `v` by `λ` without changing its direction:

```python
A * v = λ * v, where λ = 3
```

Eigenvalues characterize linear transformations and are central to PCA.

### Eigenvector

The non-zero vector `v` in `A * v = λ * v`. It points in a direction unchanged by the transformation `A`:

```python
A * v = λ * v
```

Eigenvectors identify principal directions of variance in data.

## Functions and Operators

### Maximum Function (`max(...)`)

Returns the largest value in a set:

```python
max(4, 7, 2) = 7
```

### Minimum Function (`min(...)`)

Returns the smallest value in a set:

```python
min(4, 7, 2) = 2
```

### Reciprocal (`1 / ...`)

Inverts a value:

```python
1 / 5 = 0.2
```

### Ellipsis (`...`)

Indicates continuation of a pattern:

```python
a_1 + a_2 + ... + a_n
```

## Functions and Probability

### Function Notation (`f(x)`)

Maps input `x` to an output via rule `f`:

```python
f(x) = x^2 + 2x + 1
```

### Conditional Probability Distribution (`P(x | y)`)

The probability distribution of `x` given that `y` is known:

```python
P(Output | Input)
```

Used in Bayesian inference and probabilistic models.

### Expectation Operator (`E[...]`)

The expected value (mean) of a random variable over its distribution:

```python
E[X] = Σ x_i * P(x_i)
```

### Variance (`Var(X)`)

Measures how spread out a random variable is around its mean:

```python
Var(X) = E[(X - E[X])^2]
```

### Standard Deviation (`σ(X)`)

The square root of variance — dispersion in the same units as `X`:

```python
σ(X) = sqrt(Var(X))
```

### Covariance (`Cov(X, Y)`)

Measures how two variables vary together:

```python
Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
```

Positive covariance means they increase together; negative means they move opposite.

### Correlation (`ρ(X, Y)`)

Normalized covariance, bounded in [-1, 1]. Indicates the strength and direction of a linear relationship:

```python
ρ(X, Y) = Cov(X, Y) / (σ(X) * σ(Y))
```

---

## Summary

- This section is a notation reference covering arithmetic, algebra, matrix operations, set theory, and probability used throughout the AI module.
- Norms (L1, L2, L∞) measure vector size and appear in regularization, distance calculations, and SVM margin optimization.
- The summation symbol `Σ` compacts repeated addition and underpins loss functions, means, and variances.
- Matrix operations — multiplication, transpose, inverse, determinant, trace — are the computational backbone of linear transformations and neural network forward passes.
- Eigenvalues and eigenvectors characterize directions of maximum variance and are central to PCA.
- Statistical operators (expectation, variance, covariance, correlation) quantify data spread and relationships used in model evaluation and probabilistic classifiers.

---

## Best Practices

- Return to this reference whenever an unfamiliar symbol appears in a formula — do not memorize all notation upfront.
- Standardize features before any distance-based computation (clustering, SVM, PCA) to ensure no single feature scale dominates norms.
- When computing covariance or correlation, verify that sample sizes are sufficient — estimates become unreliable with very few observations.
- Distinguish the natural log `ln(x)` (base `e`, used in cross-entropy and log-likelihood) from `log2(x)` (base 2, used in information-theoretic entropy).
- Remember that matrix inversion requires a non-zero determinant — always check this condition before attempting to solve a linear system analytically.

---

## Quiz

**Q1:** What does the L2 (Euclidean) norm of a vector measure, and where does it appear in ML algorithms?
> The L2 norm measures the straight-line length of a vector. It appears in distance calculations for k-means clustering, SVM margin maximization, and L2 regularization (weight decay).

**Q2:** What is the role of eigenvalues and eigenvectors in PCA?
> Eigenvectors define the directions (principal components) along which the data varies most; eigenvalues quantify how much variance each component explains. PCA selects the top-k eigenvectors by descending eigenvalue.

**Q3:** How does covariance differ from correlation?
> Covariance measures the joint variability of two variables and is scale-dependent. Correlation normalizes covariance by both standard deviations, producing a scale-free value in [-1, 1] that indicates strength and direction of a linear relationship.

**Q4:** What does `P(A|B)` represent in Bayesian inference?
> `P(A|B)` is the posterior probability — the probability of event A given that B has occurred. It is derived from the prior `P(A)`, the likelihood `P(B|A)`, and the marginal `P(B)` via Bayes' theorem.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-11-Principal-Component-Analysis]] — eigenvalues and eigenvectors are central to PCA
- see:: [[Section-7-Naive-Bayes]] — conditional probability notation used here
- see:: [[Section-8-Support-Vector-Machines]] — norms and vector operations underpin SVM

**Terms**
- norm, eigenvalue, eigenvector, summation, matrix multiplication, transpose, determinant, conditional probability, covariance, variance, standard deviation
