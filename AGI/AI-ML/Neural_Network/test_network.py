"""Tests for the neural network built from scratch with NumPy."""

import numpy as np
import pytest

from network import (
    Layer,
    NeuralNetwork,
    deriv_mse_loss,
    deriv_relu,
    deriv_sigmoid,
    deriv_tanh,
    mse_loss,
    relu,
    sigmoid,
    tanh,
)


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================


class TestSigmoid:
    def test_midpoint(self) -> None:
        assert sigmoid(np.array([0])) == pytest.approx(0.5, abs=1e-6)

    def test_positive_saturation(self) -> None:
        assert sigmoid(np.array([10])) == pytest.approx(1.0, abs=1e-4)

    def test_negative_saturation(self) -> None:
        assert sigmoid(np.array([-10])) == pytest.approx(0.0, abs=1e-4)

    def test_vector_input(self) -> None:
        result = sigmoid(np.array([-10, 0, 10]))
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0, abs=1e-4)
        assert result[1] == pytest.approx(0.5, abs=1e-4)
        assert result[2] == pytest.approx(1.0, abs=1e-4)


class TestDerivSigmoid:
    def test_identity_holds(self) -> None:
        x = np.array([0.0, 1.0, -1.0, 0.5])
        s = sigmoid(x)
        expected = s * (1 - s)
        assert deriv_sigmoid(x) == pytest.approx(expected, abs=1e-6)

    def test_maximum_at_zero(self) -> None:
        assert deriv_sigmoid(np.array([0])) == pytest.approx(0.25, abs=1e-6)


class TestRelu:
    def test_positive(self) -> None:
        assert relu(np.array([5])) == 5.0

    def test_negative(self) -> None:
        assert relu(np.array([-3])) == 0.0

    def test_zero(self) -> None:
        assert relu(np.array([0])) == 0.0

    def test_vector(self) -> None:
        result = relu(np.array([-2, 0, 3]))
        assert np.array_equal(result, [0, 0, 3])


class TestDerivRelu:
    def test_positive(self) -> None:
        assert deriv_relu(np.array([5])) == 1.0

    def test_negative(self) -> None:
        assert deriv_relu(np.array([-3])) == 0.0

    def test_zero(self) -> None:
        assert deriv_relu(np.array([0])) == 0.0

    def test_vector(self) -> None:
        result = deriv_relu(np.array([-2, 0, 3]))
        assert np.array_equal(result, [0, 0, 1])


class TestTanh:
    def test_zero(self) -> None:
        assert tanh(np.array([0])) == pytest.approx(0.0, abs=1e-6)

    def test_positive(self) -> None:
        assert tanh(np.array([10])) == pytest.approx(1.0, abs=1e-4)

    def test_negative(self) -> None:
        assert tanh(np.array([-10])) == pytest.approx(-1.0, abs=1e-4)


class TestDerivTanh:
    def test_identity_holds(self) -> None:
        x = np.array([0.0, 0.5, 1.0, -0.5])
        t = tanh(x)
        expected = 1 - t ** 2
        assert deriv_tanh(x) == pytest.approx(expected, abs=1e-6)

    def test_at_zero(self) -> None:
        assert deriv_tanh(np.array([0])) == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class TestMseLoss:
    def test_perfect_prediction(self) -> None:
        y_true = np.array([[1], [0], [0], [1]])
        assert mse_loss(y_true, y_true) == pytest.approx(0.0, abs=1e-10)

    def test_imperfect_prediction(self) -> None:
        y_true = np.array([[1], [0]])
        y_pred = np.array([[0.5], [0.5]])
        loss = mse_loss(y_true, y_pred)
        assert loss > 0

    def test_specific_value(self) -> None:
        y_true = np.array([[1], [0]])
        y_pred = np.array([[1], [1]])
        assert mse_loss(y_true, y_pred) == pytest.approx(0.5, abs=1e-6)


class TestDerivMseLoss:
    def test_shape(self) -> None:
        y_true = np.array([[1], [0], [0], [1]])
        y_pred = np.array([[0.9], [0.1], [0.2], [0.8]])
        grad = deriv_mse_loss(y_true, y_pred)
        assert grad.shape == (4, 1)

    def test_sign(self) -> None:
        y_true = np.array([[1], [0]])
        y_pred = np.array([[0.5], [0.5]])
        grad = deriv_mse_loss(y_true, y_pred)
        # y_pred < y_true → positive gradient for first sample
        assert grad[0, 0] < 0
        # y_pred > y_true → negative gradient for second sample
        assert grad[1, 0] > 0


# =============================================================================
# LAYER
# =============================================================================


class TestLayer:
    def test_forward_shape(self) -> None:
        layer = Layer(n_inputs=2, n_neurons=3, activation="sigmoid")
        x = np.array([[1.0, 2.0]])
        out = layer.forward(x)
        assert out.shape == (1, 3)

    def test_forward_caches_values(self) -> None:
        layer = Layer(n_inputs=2, n_neurons=2, activation="sigmoid")
        x = np.array([[1.0, 2.0]])
        layer.forward(x)
        assert layer.inputs is not None
        assert layer.z is not None
        assert layer.inputs.shape == (1, 2)
        assert layer.z.shape == (1, 2)

    def test_forward_with_known_weights(self) -> None:
        layer = Layer(n_inputs=2, n_neurons=1, activation="sigmoid")
        # Set manual weights and biases for deterministic test
        layer.weights = np.array([[0.5], [0.5]])
        layer.biases = np.array([[0.0]])
        x = np.array([[1.0, 1.0]])
        out = layer.forward(x)
        # z = 1*0.5 + 1*0.5 + 0 = 1.0 → sigmoid(1) ≈ 0.731
        assert out[0, 0] == pytest.approx(0.731058, abs=1e-5)

    def test_backward_updates_weights(self) -> None:
        layer = Layer(n_inputs=2, n_neurons=2, activation="sigmoid")
        x = np.array([[1.0, 2.0]])
        layer.forward(x)
        old_w = layer.weights.copy()
        # Backprop with a small gradient
        grad_out = np.array([[0.1, 0.1]])
        layer.backward(grad_out, learning_rate=0.1)
        # Weights should have changed
        assert not np.allclose(layer.weights, old_w)

    def test_backward_raises_without_forward(self) -> None:
        layer = Layer(n_inputs=2, n_neurons=2, activation="sigmoid")
        with pytest.raises(AssertionError):
            layer.backward(np.array([[0.1, 0.1]]), 0.1)

    def test_activations_all_work(self) -> None:
        for activation in ("sigmoid", "relu", "tanh"):
            layer = Layer(n_inputs=2, n_neurons=2, activation=activation)
            x = np.array([[1.0, -1.0]])
            out = layer.forward(x)
            assert out.shape == (1, 2)
            grad = layer.backward(np.array([[0.1, 0.1]]), 0.1)
            assert grad.shape == (1, 2)


# =============================================================================
# NEURAL NETWORK
# =============================================================================


class TestNeuralNetwork:
    def test_forward_shape(self) -> None:
        nn = NeuralNetwork(
            layer_sizes=[2, 4, 1],
            activations=["sigmoid", "sigmoid"],
        )
        x = np.array([[1.0, 2.0]])
        out = nn.forward(x)
        assert out.shape == (1, 1)

    def test_forward_batch_shape(self) -> None:
        nn = NeuralNetwork(
            layer_sizes=[2, 4, 1],
            activations=["sigmoid", "sigmoid"],
        )
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = nn.forward(x)
        assert out.shape == (3, 1)

    def test_predict_shape(self) -> None:
        nn = NeuralNetwork(
            layer_sizes=[2, 2, 1],
            activations=["sigmoid", "sigmoid"],
        )
        out = nn.predict(np.array([[1.0, 2.0]]))
        assert out.shape == (1, 1)

    def test_train_reduces_loss(self) -> None:
        nn = NeuralNetwork(
            layer_sizes=[2, 2, 1],
            activations=["sigmoid", "sigmoid"],
        )
        X = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
        y = np.array([[1], [0], [0], [1]])
        loss_history = nn.train(X, y, epochs=500, learning_rate=0.1, verbose=False)
        # Loss should have decreased from first recording to last
        assert len(loss_history) > 1
        assert loss_history[-1] < loss_history[0]

    def test_deeper_network_shape(self) -> None:
        nn = NeuralNetwork(
            layer_sizes=[2, 8, 4, 1],
            activations=["relu", "relu", "sigmoid"],
        )
        x = np.array([[1.0, 2.0]])
        out = nn.forward(x)
        assert out.shape == (1, 1)

    def test_all_three_architectures_converge(self) -> None:
        """Integration test: all 3 demo architectures should learn."""
        X = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
        y = np.array([[1], [0], [0], [1]])

        configs = [
            ([2, 2, 1], ["sigmoid", "sigmoid"], 1000, 0.1),
            ([2, 4, 1], ["sigmoid", "sigmoid"], 1000, 0.1),
            ([2, 8, 4, 1], ["relu", "relu", "sigmoid"], 2000, 0.05),
        ]

        for sizes, activations, epochs, lr in configs:
            nn = NeuralNetwork(sizes, activations)
            loss_history = nn.train(X, y, epochs=epochs, learning_rate=lr, verbose=False)
            assert loss_history[-1] < 0.25, (
                f"{'-'.join(str(s) for s in sizes)} failed to converge"
            )

    def test_predict_after_training_classifies_correctly(self) -> None:
        """After training, Emily should be ~1 and Frank should be ~0."""
        nn = NeuralNetwork(
            layer_sizes=[2, 4, 1],
            activations=["sigmoid", "sigmoid"],
        )
        X = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
        y = np.array([[1], [0], [0], [1]])
        nn.train(X, y, epochs=1000, learning_rate=0.1, verbose=False)

        emily = np.array([[-7, -3]])
        frank = np.array([[20, 2]])

        assert nn.predict(emily)[0, 0] > 0.5
        assert nn.predict(frank)[0, 0] < 0.5


# =============================================================================
# INVALID CONFIGURATIONS
# =============================================================================


class TestInvalidConfigs:
    def test_mismatched_activations(self) -> None:
        with pytest.raises(AssertionError):
            NeuralNetwork(
                layer_sizes=[2, 4, 1],
                activations=["sigmoid"],  # need 2, got 1
            )

    def test_unknown_activation(self) -> None:
        with pytest.raises(KeyError):
            Layer(n_inputs=2, n_neurons=2, activation="unknown")
