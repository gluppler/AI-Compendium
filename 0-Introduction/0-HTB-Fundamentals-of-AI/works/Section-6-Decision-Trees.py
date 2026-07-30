# Source: HackTheBox, Fundamentals of AI, Section 6 — Decision Trees
# Section file: ../2-Supervised-Learning-Algorithms/Section-6-Decision-Trees.md
# All fenced code blocks extracted verbatim in the order they appear in the note.

# ------------------------------------------------------------
# Gini Impurity
# ------------------------------------------------------------

# Gini(S) = 1 - Σ (pi)^2
# where pi is the proportion of class i in set S.

# Example: 30 class A, 20 class B (50 total)
pA = 30 / 50  # = 0.6
pB = 20 / 50  # = 0.4
Gini_S = 1 - (0.6**2 + 0.4**2)
       # = 1 - (0.36 + 0.16)
       # = 0.48

# ------------------------------------------------------------
# Entropy
# ------------------------------------------------------------

# Entropy(S) = - Σ pi * log2(pi)

# Same 30/20 split:
pA = 0.6
pB = 0.4
Entropy_S = - (0.6 * log2(0.6) + 0.4 * log2(0.4))
          # = - (0.6 * (-0.737) + 0.4 * (-1.322))
          # = 0.971

# ------------------------------------------------------------
# Information Gain
# ------------------------------------------------------------

# Information Gain(S, A) = Entropy(S) - Σ ((|Sv| / |S|) * Entropy(Sv))
# where Sv is the subset of S for each value v of feature A.

# Example: 50 instances, feature F (values 1 and 2):
#   F = 1: 30 instances, 20 class A, 10 class B
#   F = 2: 20 instances, 10 class A, 10 class B

Entropy_S = 0.971

# Entropy of subsets
Entropy_S1 = - (0.667 * log2(0.667) + 0.333 * log2(0.333))  # = 0.918
Entropy_S2 = - (0.5 * log2(0.5) + 0.5 * log2(0.5))          # = 1.0

# Weighted average
Weighted_Entropy = (30/50) * 0.918 + (20/50) * 1.0  # = 0.951

# Information gain
IG_S_F = 0.971 - 0.951  # = 0.020
