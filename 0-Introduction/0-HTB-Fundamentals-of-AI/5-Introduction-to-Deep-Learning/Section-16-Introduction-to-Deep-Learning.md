---
tags:
  - type/note
  - theme/deep-learning
  - theme/machine-learning
aliases: ["Section 16 - Introduction to Deep Learning"]
lead: Deep learning uses multi-layer neural networks to automatically learn hierarchical feature representations from raw data.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 16."
---

Deep learning is a subfield of machine learning that uses artificial neural networks with multiple layers to learn complex patterns from raw data. Where classical ML pipelines depend on hand-crafted features, deep networks discover these representations automatically through hierarchical composition — lower layers detect simple structures; higher layers combine them into abstract concepts. This makes deep learning the dominant approach for perception tasks such as image classification, speech recognition, and natural language understanding.

## Motivation Behind Deep Learning

Two complementary drivers explain the rise of deep learning:

- `Solving Complex Problems:` Many problems — image recognition, machine translation, protein folding — have input spaces too high-dimensional and non-linear for classical approaches to handle. Deep networks approximate arbitrarily complex functions given sufficient depth, width, and data.
- `Mimicking the Human Brain:` The layered architecture of ANNs is loosely inspired by cortical hierarchy in biological brains. Each layer builds a progressively abstract representation, analogous to how visual cortex progresses from edge detection to object recognition.

## Important Concepts in Deep Learning

### Artificial Neural Networks (ANNs)

`Artificial Neural Networks` (`ANNs`) are parameterized computation graphs organized into layers of interconnected `neurons`. Each edge carries a learnable `weight`; each neuron accumulates a weighted sum of its inputs, adds a `bias`, and applies a nonlinear `activation function`. The weights encode all knowledge the network has extracted from data.

### Layers

Networks are organized into three roles:

- `Input Layer:` Accepts the raw feature vector or tensor.
- `Hidden Layers:` Perform successive nonlinear transformations. Depth here is what makes a network "deep."
- `Output Layer:` Produces the prediction — a scalar for regression, a probability vector for classification.

### Activation Functions

Without nonlinearity, stacking linear layers collapses to a single affine map. Activation functions break this by introducing nonlinearity after each weighted sum. Common choices:

- `Sigmoid:` Maps $\mathbb{R}$ to $(0,1)$; suffers from vanishing gradients in deep networks.
- `ReLU (Rectified Linear Unit):` $f(x) = \max(0, x)$; sparse activations and fast convergence.
- `Tanh (Hyperbolic Tangent):` Maps $\mathbb{R}$ to $(-1,1)$; zero-centered, but still prone to saturation.

### Backpropagation

Backpropagation computes the gradient of the loss with respect to every weight by applying the chain rule layer-by-layer from output to input. These gradients are then handed to an optimizer to update the parameters.

### Loss Function

The loss quantifies the discrepancy between predictions and targets. Choice depends on the task:

- `Mean Squared Error (MSE):` Standard for regression.
- `Cross-Entropy Loss:` Standard for classification; penalizes confident wrong predictions heavily.

### Optimizer

Optimizers use the gradients from backpropagation to update weights. Key options:

- `Stochastic Gradient Descent (SGD)` — simple, noisy updates.
- `Adam` — adaptive per-parameter learning rates; default for most modern architectures.
- `RMSprop` — similar to Adam, often used in RNN training.

### Hyperparameters

Hyperparameters govern the training process and are fixed before training begins: learning rate, batch size, number of layers, neurons per layer, dropout rate. Proper tuning is critical — a misset learning rate alone can prevent convergence entirely.

---

## Summary

- Deep learning uses multi-layer artificial neural networks to automatically learn hierarchical feature representations from raw data, eliminating the need for manual feature engineering.
- ANNs consist of neurons organized in layers — input, hidden, and output — where each neuron computes a weighted sum plus bias passed through an activation function.
- Non-linear activation functions (Sigmoid, ReLU, Tanh) are essential — without them stacked layers collapse into a single linear transformation.
- Backpropagation applies the chain rule in reverse through the network to compute gradients of the loss with respect to every weight.
- Optimizers (SGD, Adam, RMSprop) use these gradients to update weights; Adam is the default for most modern architectures.
- Hyperparameters (learning rate, batch size, depth, width, dropout) must be set before training and critically affect whether the model converges.

---

## Best Practices

- Use ReLU as the default hidden-layer activation function — it avoids the vanishing gradient issue of sigmoid and tanh while being computationally efficient.
- Set the learning rate as the most impactful hyperparameter first; a misset learning rate prevents convergence even with everything else correct.
- Start with Adam optimizer when in doubt — it adapts per-parameter learning rates and converges robustly on most architectures.
- Use cross-entropy loss for classification tasks and MSE for regression — matching the loss to the task is a prerequisite for meaningful gradient signals.
- Monitor training and validation loss together — divergence between them is the primary signal of overfitting in deep networks.
- Normalize inputs before training; large input magnitudes produce large gradients that destabilize weight updates.

---

## Quiz

**Q1:** Why are non-linear activation functions necessary in deep neural networks?
> Without non-linearity, any number of stacked linear layers is equivalent to a single linear transformation. Activation functions allow networks to approximate complex, non-linear functions and separate non-linearly-separable classes.

**Q2:** What does backpropagation compute and how does it work?
> Backpropagation computes the gradient of the loss with respect to every weight in the network by applying the chain rule layer by layer from the output back to the input. These gradients indicate how much each weight contributed to the prediction error.

**Q3:** What is the difference between the loss function and the optimizer?
> The loss function quantifies the error between predictions and targets. The optimizer uses the gradients of the loss to update the network's weights — it decides how large each update step is and in which direction.

**Q4:** Name the three common activation functions and their key limitation or advantage.
> Sigmoid: maps to (0,1), prone to vanishing gradients in deep networks. ReLU: `max(0,x)`, fast convergence and no saturation for positive inputs, default choice. Tanh: maps to (-1,1), zero-centered but still saturates for large magnitudes.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-17-Perceptrons]] — the atomic building block of all deep networks
- see:: [[Section-18-Neural-Networks]] — multi-layer extension of the perceptron

**Terms**
- ANN, hidden layers, activation function, backpropagation, loss function, optimizer, hyperparameters, ReLU, sigmoid, tanh, gradient descent
