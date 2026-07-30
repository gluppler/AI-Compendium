---
tags:
  - type/note
  - theme/deep-learning
aliases: ["Section 17 - Perceptrons"]
lead: A perceptron is a single-neuron model computing a weighted sum plus bias through an activation function — limited to linearly separable problems.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 17."
---

The `perceptron` is the simplest trainable model of a neuron. It takes a vector of inputs, computes a weighted sum, shifts it by a bias, and passes the result through an activation function to produce a single output.

## Structure of a Perceptron
![[02 - Perceptrons_0.png]]

A perceptron consists of the following components:

- `Input Values (x1, x2, ..., xn):` Feature values fed into the unit.
- `Weights (w1, w2, ..., wn):` Scalar multipliers that determine each input's contribution. Negative weights suppress activation; positive weights promote it.
- `Summation Function (∑):` Computes the dot product $\sum_i w_i x_i$.
- `Bias (b):` An offset added to the weighted sum, allowing the decision boundary to shift independently of the inputs.
- `Activation Function (f):` Introduces nonlinearity and determines whether the neuron fires.
- `Output (y):` The scalar result of $f\!\left(\sum_i w_i x_i + b\right)$.

## Deciding to Play Tennis

A concrete walkthrough: predict whether to play tennis from four weather features.

- `Outlook`: Sunny (0), Overcast (1), Rainy (2)
- `Temperature`: Hot (0), Mild (1), Cool (2)
- `Humidity`: High (0), Normal (1)
- `Wind`: Weak (0), Strong (1)

Weights and bias:

- `w1` (Outlook) = 0.3
- `w2` (Temperature) = 0.2
- `w3` (Humidity) = -0.4
- `w4` (Wind) = -0.2
- `b` (Bias) = 0.1

Step activation function:

```python
def step_activation(x):
    """Step activation function."""
    return 1 if x > 0 else 0
```

Given a sunny, mild, low-humidity, calm day — inputs (0, 1, 0, 0):

Weighted sum:

$$0.3 \times 0 + 0.2 \times 1 + (-0.4) \times 0 + (-0.2) \times 0 = 0.2$$

Adding bias: $0.2 + 0.1 = 0.3$

Applying step activation: $f(0.3) = 1$ — **Play Tennis**.

Full Python implementation:

```python
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
```

## The Limitations of Perceptrons

A single-layer perceptron can only learn a linear decision boundary — a hyperplane that separates two classes in the input space. Any dataset that is not linearly separable cannot be correctly classified.

The canonical failure case is XOR: no single straight line can partition the four XOR input combinations into their two output classes. This limitation drove the move to multi-layer architectures, where stacked nonlinear layers can carve out arbitrarily complex decision regions.

---

## Summary

- A perceptron is the simplest trainable neuron model: it computes a weighted sum of inputs, adds a bias, and passes the result through an activation function to produce one output.
- Components: input values, scalar weights, summation function, bias offset, activation function, and output.
- Weights control each input's contribution — positive weights promote activation, negative weights suppress it; the bias allows the decision boundary to shift independently.
- The step activation function outputs 1 if the weighted sum exceeds a threshold and 0 otherwise, producing a binary classification.
- The tennis example demonstrates the full forward computation: multiply each feature by its weight, sum, add bias, apply activation.
- A single-layer perceptron is limited to linearly separable problems — it cannot solve XOR, which drove the development of multi-layer networks.

---

## Best Practices

- Use the perceptron as a conceptual building block for understanding neural networks — every neuron in a modern MLP is a generalized perceptron.
- Recognize the linear separability limitation: if a task is not linearly separable, a single perceptron is the wrong tool — use a multi-layer network.
- Initialize weights to small random values rather than zero; identical weights break symmetry and prevent the network from learning diverse features.
- The bias term is critical — without it the decision boundary is forced to pass through the origin, greatly restricting what can be learned.
- Prefer differentiable activation functions (sigmoid, ReLU) over the step function in practice — the step function has zero gradient almost everywhere, preventing gradient-based learning.

---

## Quiz

**Q1:** What is the mathematical operation a perceptron performs?
> It computes the weighted sum of inputs `Σ(w_i * x_i)`, adds a bias `b`, and applies an activation function `f` to produce output `y = f(Σ(w_i * x_i) + b)`.

**Q2:** What is the role of the bias in a perceptron?
> The bias shifts the decision boundary independently of the input values. Without it, the boundary must pass through the origin, preventing the perceptron from correctly classifying many datasets.

**Q3:** Why can a single-layer perceptron not solve the XOR problem?
> XOR is not linearly separable — no single hyperplane can partition the four input combinations into their correct output classes. The perceptron's decision boundary is always a linear hyperplane.

**Q4:** What drove the development of multi-layer networks after the perceptron?
> The discovery that single-layer perceptrons can only learn linearly separable functions. Stacking multiple layers of neurons with non-linear activations allows the network to learn arbitrarily complex decision boundaries.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-18-Neural-Networks]] — MLPs overcome the XOR limitation by stacking layers
- see:: [[Section-2-Mathematics-Refresher-for-AI]] — weighted sums and activation use matrix/vector notation

**Terms**
- perceptron, weights, bias, activation function, step function, linear separability, XOR problem, weighted sum
