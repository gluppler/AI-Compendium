"""
A modular neural network built entirely from scratch using only NumPy.

No TensorFlow, PyTorch, or any ML framework — just raw matrix math and
the chain rule from calculus. Every concept is explained inline.

Architecture: fully connected (dense) feedforward network with
configurable layer sizes, activations, and training hyperparameters.

Usage:
    python network.py
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================
#
# Activations introduce non-linearity. Without them, stacking linear layers
# would just equal one big linear layer — the network could never learn
# complex patterns. Each activation has:
#   - fn(x):   the forward function
#   - deriv(x): its derivative (needed for backpropagation)
#
# Why we need derivatives: backprop uses the chain rule, which multiplies
# gradients together. The derivative of the activation is one link in that
# chain. We cache the pre-activation value (z) during forward and pass it
# into deriv() during backward.

# -----------------------------------------------------------------------------
# Sigmoid
# -----------------------------------------------------------------------------
# Formula:  sigmoid(x) = 1 / (1 + e^(-x))
# Range:    (0, 1)  — outputs a probability-like value
# Derivative: sigmoid(x) * (1 - sigmoid(x))
#
# Use case: binary classification (output layer), or small toy networks.
# Limitation: saturates at extreme values → vanishing gradients.


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Squashes any real number into the range (0, 1).
    Large positive x → close to 1. Large negative x → close to 0.
    """
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the sigmoid function.
    Uses the identity: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Returns x if x > 0, otherwise 0.
    Simple, fast, and works surprisingly well in practice.
    """
    return np.maximum(0, x)


def deriv_relu(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU: 1 where x > 0, 0 elsewhere.
    """
    return (x > 0).astype(float)


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Squashes any real number into the range (-1, 1).
    Zero-centered — outputs can be positive or negative.
    """
    return np.tanh(x)


def deriv_tanh(x: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh: 1 - tanh(x)^2
    """
    t = tanh(x)
    return 1 - t**2


# Map activation names (strings) to their (fn, deriv_fn) pairs.
# This lets us specify activations as strings like 'sigmoid' or 'relu'
# when building a network architecture.
ActivationPair = tuple[
    Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]
]

ACTIVATIONS: dict[str, ActivationPair] = {
    "sigmoid": (sigmoid, deriv_sigmoid),
    "relu": (relu, deriv_relu),
    "tanh": (tanh, deriv_tanh),
}


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
#
# The loss function measures how wrong the network's predictions are.
# Training = finding weights that minimize this loss.

# -----------------------------------------------------------------------------
# Mean Squared Error (MSE)
# -----------------------------------------------------------------------------
# Formula:  MSE = (1/n) * sum((y_true - y_pred)^2)
#
# Simple and widely used for regression tasks. Penalizes large errors more
# than small ones (quadratic penalty).


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Returns the average squared difference between predictions and targets.
    Lower is better. A perfect score is 0.
    """
    return float(((y_true - y_pred) ** 2).mean())


def deriv_mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of MSE w.r.t. y_pred:
    d(MSE)/d(y_pred) = (2/n) * (y_pred - y_true)

    This is the gradient that starts the backprop chain.
    """
    n = y_true.shape[0]
    return 2 * (y_pred - y_true) / n


# -----------------------------------------------------------------------------
# Binary Cross-Entropy (BCE)
# -----------------------------------------------------------------------------
# Formula: BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
#
# Preferred for binary classification (works better than MSE with sigmoid).


def bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary cross-entropy — standard loss for binary classification.
    Values are clipped to avoid log(0) = -inf.
    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def deriv_bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of BCE w.r.t. y_pred:
    d(BCE)/d(y_pred) = (y_pred - y_true) / (y_pred * (1 - y_pred))
    But when paired with sigmoid, this simplifies (see cross-entropy + sigmoid
    gradient derivation).
    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]


# =============================================================================
# LAYER CLASS
# =============================================================================
#
# A single fully-connected (dense) layer. It:
#   1. Stores its own weights and biases (initialized randomly)
#   2. Computes forward pass: output = activation(input @ weights + biases)
#   3. Computes backward pass: calculates gradients, updates weights,
#      and passes the gradient back to the previous layer


class Layer:
    """
    A single dense layer in the neural network.

    During forward(), it computes Z = inputs @ weights + biases, then applies
    the activation function. It caches the inputs and pre-activation values
    so backward() can use them.

    During backward(), it receives the gradient of the loss with respect to
    its output, uses the cached values to compute weight/biases gradients,
    updates the parameters, and returns the gradient for the previous layer.
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        activation: str = "sigmoid",
    ) -> None:
        """
        Set up a layer with random initial weights and zero biases.

        Parameters
        ----------
        n_inputs : int
            Number of input features (or neurons from previous layer).
        n_neurons : int
            Number of neurons in this layer (output dimension).
        activation : str
            Activation function name — 'sigmoid', 'relu', or 'tanh'.

        Weights are initialized with a simple random normal distribution.
        In practice, more sophisticated init (Xavier, He) is used, but
        random normal is fine for educational purposes.
        """
        self.activation_name: str = activation
        fn, deriv_fn = ACTIVATIONS[activation]
        self.fn: Callable[[np.ndarray], np.ndarray] = fn
        self.deriv_fn: Callable[[np.ndarray], np.ndarray] = deriv_fn

        # Weights: shape (n_inputs, n_neurons)
        # Each column holds all weights going into one neuron.
        # Small random values to break symmetry (all zeros = all neurons
        # learn the same thing).
        #
        # The initialization scale matters! If weights start too large,
        # sigmoid/tanh saturate (gradients vanish). If too small with ReLU,
        # neurons can die (always output 0).
        #
        # We use two common strategies:
        #   - Xavier/Glorot init  (for sigmoid, tanh): scale = sqrt(1 / n_inputs)
        #     Keeps the variance of activations roughly constant across layers.
        #   - He init  (for ReLU): scale = sqrt(2 / n_inputs)
        #     Accounts for ReLU zeroing out half the neurons, so variance
        #     needs a larger boost.
        if activation in ("sigmoid", "tanh"):
            scale = np.sqrt(1.0 / n_inputs)
        elif activation == "relu":
            scale = np.sqrt(2.0 / n_inputs)
        else:
            scale = 0.5
        self.weights: np.ndarray = np.random.randn(n_inputs, n_neurons) * scale

        # Biases: shape (1, n_neurons)
        # One bias per neuron. Initialized to zero — symmetry breaking
        # is handled by the random weights.
        self.biases: np.ndarray = np.zeros((1, n_neurons))

        # Cached values for backpropagation
        self.inputs: np.ndarray | None = None
        self.z: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the output of this layer given inputs.

        Forward pass formula:
            Z = inputs @ weights + biases
            output = activation(Z)

        We cache 'inputs' and 'z' because backward() needs them to
        compute gradients.

        Parameters
        ----------
        inputs : ndarray of shape (batch_size, n_inputs)

        Returns
        -------
        ndarray of shape (batch_size, n_neurons)
        """
        self.inputs = inputs
        self.z = inputs @ self.weights + self.biases
        return self.fn(self.z)

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Compute gradients and update weights using gradient descent.

        This is where the chain rule is applied:
            dL/dW = (dL/doutput) * (doutput/dZ) * (dZ/dW)
                   = grad_output * deriv_activation(Z) * inputs^T

        Step by step:
        1. dL/dZ = grad_output * deriv_activation(z)
           (element-wise: how much does the loss change with respect to
            the pre-activation value?)
        2. dL/dW = inputs^T @ dL/dZ
           (outer product: how much does each weight contribute?)
        3. dL/db = sum(dL/dZ, axis=0)
           (sum across batch: each bias affects all samples)
        4. Update: W -= learning_rate * dL/dW, b -= learning_rate * dL/db
        5. Return dL/d(inputs) = dL/dZ @ weights^T
           (propagate gradient to the previous layer)

        Parameters
        ----------
        grad_output : ndarray of shape (batch_size, n_neurons)
            Gradient of loss with respect to this layer's output.
            Comes from the next layer (or the loss function for the
            output layer).
        learning_rate : float
            Step size for gradient descent.

        Returns
        -------
        ndarray of shape (batch_size, n_inputs)
            Gradient of loss with respect to this layer's input,
            to be passed backward to the previous layer.
        """
        assert self.z is not None, "Must call forward() before backward()"
        assert self.inputs is not None, "Must call forward() before backward()"

        # Step 1: gradient through the activation function
        # dL/dZ = dL/d(output) * d(activation)/dZ  (element-wise)
        dZ = grad_output * self.deriv_fn(self.z)

        # Step 2: gradient with respect to weights
        # dL/dW = (inputs^T) @ dL/dZ
        # Each element (i,j) = how much does weight[i,j] affect the loss?
        dW = self.inputs.T @ dZ

        # Step 3: gradient with respect to biases
        # dL/db = sum(dL/dZ across all samples in batch)
        # Keepdims=True so it stays shape (1, n_neurons) matching self.biases
        db = np.sum(dZ, axis=0, keepdims=True)

        # Step 4: gradient descent update
        # Move weights in the opposite direction of the gradient
        # to reduce the loss.
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db

        # Step 5: return gradient for the previous layer
        # dL/d(inputs) = dL/dZ @ weights^T
        # This tells the previous layer how its output affected the loss.
        grad_input = dZ @ self.weights.T

        return grad_input


# =============================================================================
# NEURAL NETWORK CLASS
# =============================================================================
#
# Orchestrates multiple layers: passes data forward, backpropagates error,
# and runs the training loop.


class NeuralNetwork:
    """
    A fully-connected feedforward neural network.

    Parameters
    ----------
    layer_sizes : list of int
        Number of neurons in each layer, including input and output.
        Example: [2, 3, 1] means 2 inputs → 3 hidden → 1 output.
    activations : list of str
        Activation for each layer (except the input layer which has none).
        Must be len(layer_sizes) - 1.
        Example: ['sigmoid', 'sigmoid'] for a 2-3-1 network.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str],
    ) -> None:
        """
        Build the network by creating Layer objects.

        For each adjacent pair (layer_sizes[i], layer_sizes[i+1]),
        we create one Layer with the corresponding activation.
        """
        assert (
            len(activations) == len(layer_sizes) - 1
        ), "Must have one activation per layer (excluding input)"

        self.layers: list[Layer] = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                n_inputs=layer_sizes[i],
                n_neurons=layer_sizes[i + 1],
                activation=activations[i],
            )
            self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Pass input data through all layers sequentially.

        The output of each layer becomes the input to the next.
        This is called a "forward pass."

        Parameters
        ----------
        X : ndarray of shape (batch_size, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (batch_size, output_size)
            Network's predictions.
        """
        out: np.ndarray = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        learning_rate: float,
    ) -> None:
        """
        Backpropagate the error through all layers.

        Starts from the output layer's gradient (derivative of loss)
        and moves backward through the network, calling each layer's
        backward() method. This implements the chain rule.

        Parameters
        ----------
        y_true : ndarray of shape (batch_size, 1)
            Ground truth labels.
        y_pred : ndarray of shape (batch_size, 1)
            Network's predictions.
        learning_rate : float
            Step size for gradient descent.
        """
        grad = deriv_mse_loss(y_true, y_pred)

        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.1,
        verbose: bool = True,
    ) -> list[float]:
        """
        Train the network using gradient descent.

        Training loop:
        1. Forward pass: compute predictions
        2. Loss calculation: measure error
        3. Backward pass: compute gradients via backpropagation
        4. Parameter update: gradient descent step

        Repeat for the specified number of epochs.

        Parameters
        ----------
        X : ndarray of shape (batch_size, n_features)
            Training data.
        y : ndarray of shape (batch_size, 1)
            Target labels.
        epochs : int
            Number of complete passes through the training data.
        learning_rate : float
            Step size for gradient descent.
        verbose : bool
            If True, print loss every 100 epochs.

        Returns
        -------
        list[float]
            Loss values recorded every 100 epochs (for testing / plotting).
        """
        loss_history: list[float] = []
        for epoch in range(epochs):
            y_pred = self.forward(X)

            self.backward(y, y_pred, learning_rate)

            if epoch % 100 == 0:
                loss = mse_loss(y, y_pred)
                loss_history.append(loss)
                if verbose:
                    print(f"  Epoch {epoch:4d}  loss: {loss:.6f}")

        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for new data.

        Parameters
        ----------
        X : ndarray of shape (batch_size, n_features)

        Returns
        -------
        ndarray of shape (batch_size, output_size)
        """
        return self.forward(X)


if __name__ == "__main__":
    # =========================================================================
    # DATASET
    # =========================================================================
    #
    # Toy dataset: predicting gender from (normalized) weight and height.
    #
    # Why these values look odd: they're normalized (zero-centered, small
    # scale) for educational purposes. Real data would be on different scales.
    #
    # Features:
    #   Column 0: weight (normalized)
    #   Column 1: height (normalized)
    #
    # Target:
    #   1 = female, 0 = male

    data = np.array(
        [
            [-2, -1],  # Alice   — female
            [25, 6],  # Bob     — male
            [17, 4],  # Charlie — male
            [-15, -6],  # Diana   — female
        ]
    )

    # Targets need shape (n, 1) for proper matrix math
    all_y_trues = np.array(
        [[1], [0], [0], [1]]  # Alice   — female  # Bob     — male  # Charlie — male
    )  # Diana   — female

    # Unseen test samples to evaluate after training
    emily = np.array([[-7, -3]])  # should predict ~1 (female)
    frank = np.array([[20, 2]])  # should predict ~0 (male)
    # =========================================================================
    # DEMO 1: Original Architecture  (2 inputs → 2 hidden → 1 output)
    # =========================================================================
    # This matches Victor Zhou's original tutorial exactly:
    #   - 1 hidden layer with 2 neurons
    #   - Sigmoid activation on all layers
    #   - Trained for 1000 epochs at learning rate 0.1
    #
    # A 2-2-1 network is the smallest possible usable network. The hidden layer
    # can only learn 2 features, which is barely enough for this simple dataset.
    # It usually converges, but loss can be noisy due to random initialization.

    print("=" * 65)
    print("DEMO 1: 2-2-1  (original architecture, sigmoid all layers)")
    print("=" * 65)

    # Build
    #   Input layer:  2 neurons (weight, height)
    #   Hidden layer: 2 neurons
    #   Output layer: 1 neuron  (gender probability)
    nn1 = NeuralNetwork(
        layer_sizes=[2, 2, 1],
        activations=["sigmoid", "sigmoid"],
    )

    # Train
    print("\nTraining...")
    nn1.train(data, all_y_trues, epochs=1000, learning_rate=0.1)

    # Predict
    pred_emily = nn1.predict(emily)[0, 0]
    pred_frank = nn1.predict(frank)[0, 0]
    print(f"\nEmily  (expected ~1.0):  {pred_emily:.4f}")
    print(f"Frank  (expected ~0.0):  {pred_frank:.4f}")

    # Interpretation: results close to 1.0 (Emily) and 0.0 (Frank) mean the
    # network learned to distinguish the two classes. With only 2 hidden
    # neurons, expect some variance between runs.

    # =========================================================================
    # DEMO 2: Wider Hidden Layer  (2 inputs → 4 hidden → 1 output)
    # =========================================================================
    # Same structure as Demo 1, but the hidden layer has 4 neurons instead of 2.
    # More neurons = more capacity to learn patterns = potentially faster/better
    # convergence. Still uses sigmoid everywhere.

    print("\n")
    print("=" * 65)
    print("DEMO 2: 2-4-1  (wider hidden layer, sigmoid all layers)")
    print("=" * 65)

    # Build
    #   Input layer:  2 neurons
    #   Hidden layer: 4 neurons (double the capacity of Demo 1)
    #   Output layer: 1 neuron
    nn2 = NeuralNetwork(
        layer_sizes=[2, 4, 1],
        activations=["sigmoid", "sigmoid"],
    )

    # Train
    print("\nTraining...")
    nn2.train(data, all_y_trues, epochs=1000, learning_rate=0.1)

    # Predict
    pred_emily = nn2.predict(emily)[0, 0]
    pred_frank = nn2.predict(frank)[0, 0]
    print(f"\nEmily  (expected ~1.0):  {pred_emily:.4f}")
    print(f"Frank  (expected ~0.0):  {pred_frank:.4f}")

    # Interpretation: with 4 hidden neurons, the network typically converges
    # faster and more reliably than the 2-neuron version. More parameters
    # = more expressive power, but also more computation.

    # =========================================================================
    # DEMO 3: Deeper Network with ReLU  (2 → 8 → 4 → 1)
    # =========================================================================
    # Now we get interesting:
    #   - 2 hidden layers (depth!): 8 neurons, then 4 neurons
    #   - ReLU activation on hidden layers (not sigmoid!)
    #   - Sigmoid on the output layer (needed for binary classification)
    #
    # Why ReLU for hidden layers?
    #   - Helps with vanishing gradient (derivative = 1 for positive values)
    #   - Computationally cheap (just max with 0)
    #   - Often converges faster than sigmoid in deeper nets
    #
    # Why sigmoid on the output?
    #   - We need a value in (0, 1) for binary classification probability
    #   - ReLU can output arbitrarily large positive values (not probabilities)
    #   - Tanh outputs (-1, 1), not (0, 1)
    #   - Only sigmoid gives us a clean probability estimate
    #
    # Why deeper?
    #   - Multiple hidden layers can learn hierarchical features
    #   - First layer learns simple patterns, later layers combine them
    #   - With 2 hidden layers, the network can learn more complex decision
    #     boundaries — though this tiny dataset doesn't need it

    print("\n")
    print("=" * 65)
    print("DEMO 3: 2-8-4-1  (deeper network, ReLU hidden layers)")
    print("=" * 65)

    # Build
    #   Input layer:   2 neurons
    #   Hidden layer 1: 8 neurons with ReLU
    #   Hidden layer 2: 4 neurons with ReLU
    #   Output layer:  1 neuron with sigmoid
    nn3 = NeuralNetwork(
        layer_sizes=[2, 8, 4, 1],
        activations=["relu", "relu", "sigmoid"],
    )

    # Train
    #   More epochs because deeper networks sometimes need more time to converge
    print("\nTraining...")
    nn3.train(data, all_y_trues, epochs=2000, learning_rate=0.05)

    # Predict
    pred_emily = nn3.predict(emily)[0, 0]
    pred_frank = nn3.predict(frank)[0, 0]
    print(f"\nEmily  (expected ~1.0):  {pred_emily:.4f}")
    print(f"Frank  (expected ~0.0):  {pred_frank:.4f}")

    # Interpretation: this is overkill for 4 training samples (more parameters
    # than data points!), but it demonstrates that the same code handles deeper
    # architectures seamlessly. With ReLU, watch for dead neurons (all zeros)
    # if the learning rate is too high.

    # =========================================================================
    # SUMMARY
    # =========================================================================
    #
    # What you just watched three networks learn the same task with different
    # architectures:
    #
    #   1.  2-2-1  sigmoid/sigmoid    —  original, minimal network
    #   2.  2-4-1  sigmoid/sigmoid    —  wider, more capacity
    #   3.  2-8-4-1  relu/relu/sigmoid — deeper, modern activation
    #
    # Key takeaways:
    #   - More neurons = more capacity (but also more overfitting risk)
    #   - More layers = hierarchical learning (but also harder to train)
    #   - Activation choice matters: sigmoid for output probabilities,
    #     ReLU for hidden layers in deeper networks
    #   - Backpropagation is just the chain rule applied systematically
    #   - Everything boils down to matrix multiplications and element-wise
    #     operations — which GPUs excel at
    #
    # To experiment further:
    #   - Try different learning rates (0.01, 0.5, 1.0)
    #   - Swap sigmoid for tanh in the hidden layers
    #   - Add more training data or try the XOR problem
    #   - Implement mini-batch gradient descent
    #   - Add L2 regularization to prevent overfitting
    #   - Try the BCE loss instead of MSE (better for binary classification)
