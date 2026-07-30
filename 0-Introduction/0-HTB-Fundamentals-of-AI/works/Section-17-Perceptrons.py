# Source: HackTheBox, Fundamentals of AI, Section 17 — Perceptrons
# Section file: ../5-Introduction-to-Deep-Learning/Section-17-Perceptrons.md
# All fenced code blocks extracted verbatim in the order they appear in the note.

# ------------------------------------------------------------
# Step activation function
# ------------------------------------------------------------

def step_activation(x):
    """Step activation function."""
    return 1 if x > 0 else 0

# ------------------------------------------------------------
# Full perceptron forward pass — tennis prediction example
# ------------------------------------------------------------

# Input features
outlook = 0
temperature = 1
humidity = 0
wind = 0

# Weights and bias
w1 = 0.3
w2 = 0.2
w3 = -0.4
w4 = -0.2
b = 0.1

# Calculate weighted sum
weighted_sum = (w1 * outlook) + (w2 * temperature) + (w3 * humidity) + (w4 * wind)

# Add bias
total_input = weighted_sum + b

# Apply activation function
output = step_activation(total_input)
print(f"Output: {output}")  # Output: 1 (Play Tennis)
