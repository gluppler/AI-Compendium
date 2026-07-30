---
tags:
  - type/note
  - theme/deep-learning
aliases: ["Section 18 - Neural Networks"]
lead: Multi-layer perceptrons stack hidden layers to overcome the XOR limitation, trained via backpropagation through gradient descent.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 18."
---

![[neural_network.png]]

A `multi-layer perceptron` (`MLP`) extends the single perceptron into a directed acyclic graph of neurons organized into three stages: an input layer, one or more hidden layers, and an output layer. The key departure from the single-layer perceptron is depth — multiple nonlinear transformations compose into functions that can represent arbitrarily complex mappings.

## Neurons

A `neuron` computes $a = f(w^\top x + b)$, where $f$ is any differentiable activation function. Unlike the step-function perceptron, neurons in an MLP commonly use `sigmoid`, `ReLU`, or `tanh`, producing continuous outputs and enabling gradient-based learning.

## Input Layer
![[03 - Neural Networks_0.png]]

The `input layer` holds one node per input feature. It performs no computation — it simply passes the raw data forward into the first hidden layer.

## Hidden Layers
![[03 - Neural Networks_1.png]]

Each neuron in a `hidden layer`:

1. Receives the full output vector of the preceding layer.
2. Computes a weighted sum of those inputs.
3. Adds a bias.
4. Applies a nonlinear activation function.

Stacking multiple hidden layers allows the network to build increasingly abstract representations: early layers detect simple patterns; later layers compose those into higher-order structures. This hierarchical feature learning is what enables MLPs to solve non-linearly separable problems like XOR.

## Output Layer
![[03 - Neural Networks_2.png]]

The `output layer` produces the network's predictions. Its structure depends on the task:

- Binary classification: one neuron with sigmoid activation.
- Multi-class classification: one neuron per class, typically with softmax.
- Regression: one or more neurons with linear activation.

## The Power of Multiple Layers

MLPs overcome single-layer limitations through two mechanisms. Non-linear activation functions allow each layer to transform its input space in ways a linear map cannot. Depth then lets these transformations compose: the universal approximation theorem guarantees that an MLP with at least one sufficiently wide hidden layer can approximate any continuous function on a compact domain — with appropriate depth reducing the required width dramatically in practice.

## Activation Functions

### Types of Activation Functions

- `Sigmoid:` $\sigma(x) = \frac{1}{1+e^{-x}}$, output in $(0,1)$. Prone to vanishing gradients in deep networks.
- `ReLU (Rectified Linear Unit):` $f(x) = \max(0, x)$. Computationally cheap, avoids saturation for positive inputs, and remains the default choice for hidden layers.
- `Tanh (Hyperbolic Tangent):` Output in $(-1,1)$, zero-centered. Better gradient flow than sigmoid but still saturates.
- `Softmax:` Normalizes a vector of raw scores into a probability distribution over $K$ classes; used exclusively in the output layer for multi-class problems.

## Training MLPs

### Backpropagation
![[03 - Neural Networks_3.png]]

`Backpropagation` computes gradients of the loss with respect to every weight by applying the chain rule in reverse through the network:

1. `Forward Pass:` Propagate input through all layers to produce a prediction.
2. `Calculate Error:` Evaluate the loss function comparing the prediction to the target.
3. `Backward Pass:` Starting at the output layer, apply the chain rule to compute $\frac{\partial \mathcal{L}}{\partial w}$ for each weight.
4. `Update Weights and Biases:` Apply the optimizer update rule using the computed gradients.

### Gradient Descent
![[03 - Neural Networks_4.png]]

`Gradient descent` iteratively moves the parameters in the direction of steepest loss decrease:

$$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}$$

where $\eta$ is the learning rate. The process:

1. `Initialize Weights and Biases:` Sample from a small random distribution to break symmetry.
2. `Calculate Gradient:` Run backpropagation to get $\frac{\partial \mathcal{L}}{\partial w}$ for all parameters.
3. `Update Weights and Biases:` Apply the update rule; $\eta$ controls step size.
4. `Repeat:` Continue until the loss converges or a maximum iteration count is reached.

Backpropagation supplies the gradients; gradient descent uses them to move parameters toward a loss minimum. Together they form the core training loop for all deep networks.

---

## Summary

- Multi-layer perceptrons (MLPs) extend single perceptrons with one or more hidden layers, enabling learning of arbitrarily complex non-linear functions.
- The input layer passes raw features forward; hidden layers apply successive non-linear transformations; the output layer produces predictions.
- ReLU is the preferred hidden-layer activation for its computational efficiency and avoidance of saturation; softmax is used in the output layer for multi-class problems.
- Backpropagation computes loss gradients for every weight using the chain rule applied in reverse from output to input.
- Gradient descent updates weights by moving them in the direction of steepest loss decrease: `w ← w - η * ∂L/∂w`.
- The universal approximation theorem guarantees an MLP with at least one sufficiently wide hidden layer can approximate any continuous function on a compact domain.

---

## Best Practices

- Use ReLU as the default hidden-layer activation — it avoids vanishing gradients and is computationally cheap compared to sigmoid or tanh.
- Initialize weights with small random values (e.g., Xavier or He initialization) to break symmetry; zero-initialization causes all neurons to learn the same features.
- Pair backpropagation with an adaptive optimizer (Adam) rather than vanilla SGD to benefit from per-parameter learning rates.
- Monitor the gradient magnitudes during training — vanishing gradients in early layers indicate the need for architecture changes (residual connections, batch normalization).
- Use softmax exclusively in the output layer for multi-class classification; in hidden layers it creates winner-take-all dynamics that suppress gradient flow.
- Batch normalization between layers stabilizes training and reduces sensitivity to the initial learning rate.

---

## Quiz

**Q1:** What is the universal approximation theorem and what does it imply for MLP design?
> It states that an MLP with at least one sufficiently wide hidden layer can approximate any continuous function on a compact domain. In practice, depth reduces the required width — deeper networks are more parameter-efficient than very wide shallow ones.

**Q2:** Describe the four steps of the backpropagation algorithm.
> 1) Forward pass: propagate input through layers to produce a prediction. 2) Calculate error: evaluate the loss function. 3) Backward pass: apply the chain rule layer-by-layer from output to input to compute `∂L/∂w` for every weight. 4) Update weights: apply the gradient descent rule using the computed gradients.

**Q3:** What is the gradient descent update rule and what does the learning rate control?
> `w ← w - η * ∂L/∂w`. The learning rate `η` controls the step size — large values converge quickly but risk overshooting a minimum; small values are stable but slow.

**Q4:** Why does the output layer use different activation functions for classification vs. regression?
> For binary classification: sigmoid maps output to (0,1) for a probability. For multi-class: softmax normalizes scores into a class probability distribution. For regression: linear activation produces unbounded continuous values. Matching activation to task is required for the loss function to be interpretable.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/0-Introduction/0-HTB-Fundamentals-of-AI/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-17-Perceptrons]] — perceptrons are the atomic unit
- see:: [[Section-19-Convolutional-Neural-Networks]] — CNNs specialise the MLP for spatial data
- see:: [[Section-20-Recurrent-Neural-Networks]] — RNNs specialise for sequential data

**Terms**
- MLP, multi-layer perceptron, backpropagation, gradient descent, hidden layer, forward pass, weight initialisation, vanishing gradient
