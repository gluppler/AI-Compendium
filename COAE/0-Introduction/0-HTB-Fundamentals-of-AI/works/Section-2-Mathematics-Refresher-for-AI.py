# Source: HackTheBox, Fundamentals of AI, Section 2 — Mathematics Refresher for AI
# Section file: ../1-Introduction-to-Machine-Learning/Section-2-Mathematics-Refresher-for-AI.md
# All fenced code blocks extracted verbatim in the order they appear in the note.

# ------------------------------------------------------------
# Basic Arithmetic Operations
# ------------------------------------------------------------

# Multiplication
3 * 4  # = 12

# Division
10 / 2  # = 5

# Addition
5 + 3  # = 8

# Subtraction
9 - 4  # = 5

# ------------------------------------------------------------
# Algebraic Notations
# ------------------------------------------------------------

# Subscript notation — x_t is the value of x at time t
x_t = q(x_t | x_{t-2})

# Superscript notation — exponentiation
x**2  # = x * x

# L2 (Euclidean) norm
||v|| = sqrt(v_1**2 + v_2**2 + ... + v_n**2)

# L1 and L-infinity norms
||v||_1 = |v_1| + |v_2| + ... + |v_n|
||v||_∞ = max(|v_1|, |v_2|, ..., |v_n|)

# Summation symbol — a_1 + a_2 + ... + a_n
Σ_{i=1}^{n} a_i

# ------------------------------------------------------------
# Logarithms and Exponentials
# ------------------------------------------------------------

import math

math.log2(8)   # = 3
math.log(math.e**2)  # ln(e^2) = 2
math.e**2      # ≈ 7.389
2**3           # = 8

# ------------------------------------------------------------
# Matrix and Vector Operations
# ------------------------------------------------------------

# Matrix-vector multiplication
A * v  # [[1, 2], [3, 4]] * [5, 6] = [17, 39]

# Matrix-matrix multiplication
A * B  # [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]

# Transpose
A   = [[1, 2], [3, 4]]
A_T = [[1, 3], [2, 4]]  # A^T

# Matrix inverse
A      = [[1, 2], [3, 4]]
A_inv  = [[-2, 1], [1.5, -0.5]]  # A^{-1}

# Determinant
A      = [[1, 2], [3, 4]]
det_A  = 1 * 4 - 2 * 3  # = -2

# Trace
A      = [[1, 2], [3, 4]]
tr_A   = 1 + 4  # = 5

# ------------------------------------------------------------
# Set Theory
# ------------------------------------------------------------

S = {1, 2, 3, 4, 5}
len(S)  # |S| = 5

A = {1, 2, 3}; B = {3, 4, 5}
A | B  # A ∪ B = {1, 2, 3, 4, 5}
A & B  # A ∩ B = {3}

U = {1, 2, 3, 4, 5}; A = {1, 2, 3}
A_c = U - A  # A^c = {4, 5}

# ------------------------------------------------------------
# Comparison Operators
# ------------------------------------------------------------

a >= b  # greater than or equal to
a <= b  # less than or equal to
a == b  # equality
a != b  # inequality

# ------------------------------------------------------------
# Eigenvalues and Eigenvectors
# ------------------------------------------------------------

# A * v = λ * v, where λ = 3
A * v  # = λ * v

# ------------------------------------------------------------
# Functions and Operators
# ------------------------------------------------------------

max(4, 7, 2)  # = 7
min(4, 7, 2)  # = 2
1 / 5         # = 0.2

# Ellipsis — continuation of a pattern
a_1 + a_2 + ... + a_n

# ------------------------------------------------------------
# Functions and Probability
# ------------------------------------------------------------

# Function notation
f = lambda x: x**2 + 2*x + 1

# Conditional probability distribution
# P(Output | Input)

# Expectation operator
# E[X] = Σ x_i * P(x_i)

# Variance
# Var(X) = E[(X - E[X])^2]

# Standard deviation
# σ(X) = sqrt(Var(X))

# Covariance
# Cov(X, Y) = E[(X - E[X])(Y - E[Y])]

# Correlation
# ρ(X, Y) = Cov(X, Y) / (σ(X) * σ(Y))
