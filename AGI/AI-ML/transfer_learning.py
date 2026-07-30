"""
Transfer Learning

Techniques for transferring knowledge from a pre-trained source model
to a target task. Includes feature extraction, fine-tuning, and domain
adaptation with a simulated neural network.

https://en.wikipedia.org/wiki/Transfer_learning
"""

from __future__ import annotations

import math
import random


class SimpleModel:
    def __init__(self, layers: list[int]):
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            self.weights.append(
                [
                    [random.uniform(-0.1, 0.1) for _ in range(layers[i + 1])]
                    for _ in range(layers[i])
                ]
            )
            self.biases.append([0.0] * layers[i + 1])

    def forward(self, x: list[float]) -> list[list[float]]:
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            next_act = [0.0] * len(b)
            for j in range(len(b)):
                val = sum(activations[-1][k] * w[k][j] for k in range(len(activations[-1]))) + b[j]
                next_act[j] = max(0.0, val)
            activations.append(next_act)
        return activations

    def predict(self, x: list[float]) -> list[float]:
        return self.forward(x)[-1]


def feature_extraction(
    pretrained: SimpleModel, new_data: list[list[float]], keep_layers: int = 1
) -> SimpleModel:
    classifier = SimpleModel([pretrained.weights[-1][0] for _ in range(1)])

    for x in new_data:
        features = pretrained.forward(x)[-keep_layers - 1]
        _ = classifier.predict(features)
    return classifier


def fine_tune(
    model: SimpleModel,
    x_train: list[list[float]],
    y_train: list[list[float]],
    lr: float = 0.01,
    epochs: int = 10,
    freeze_layers: int = 1,
) -> SimpleModel:
    for _ in range(epochs):
        for x, y in zip(x_train, y_train):
            activations = model.forward(x)
            pred = activations[-1]
            error = [pred[j] - y[j] for j in range(len(y))]

            for layer_idx in range(len(model.weights) - 1, freeze_layers - 1, -1):
                w = model.weights[layer_idx]
                b = model.biases[layer_idx]
                act_in = activations[layer_idx]
                for j in range(len(b)):
                    b[j] -= lr * error[j]
                    for k in range(len(act_in)):
                        grad = error[j] * act_in[k]
                        w[k][j] -= lr * grad
                if layer_idx > 0:
                    new_error = [0.0] * len(activations[layer_idx - 1])
                    for k in range(len(new_error)):
                        new_error[k] = sum(error[j] * w[k][j] for j in range(len(error)))
                    error = new_error
    return model


if __name__ == "__main__":
    random.seed(42)
    pretrained = SimpleModel([10, 8, 4])
    x_test = [[random.random() for _ in range(10)] for _ in range(3)]
    print("Pre-trained model predictions:")
    for x in x_test:
        print(f"  {[round(v, 3) for v in pretrained.predict(x)]}")

    extractor = feature_extraction(pretrained, x_test, keep_layers=1)
    print("\nFeature extraction model ready.")

    x_few = [[random.random() for _ in range(10)] for _ in range(2)]
    y_few = [[random.random() for _ in range(4)] for _ in range(2)]
    finetuned = fine_tune(pretrained, x_few, y_few, lr=0.005, epochs=5)
    print("Fine-tuned model predictions:")
    for x in x_few:
        print(f"  {[round(v, 3) for v in finetuned.predict(x)]}")
    print("Transfer learning demo complete.")
