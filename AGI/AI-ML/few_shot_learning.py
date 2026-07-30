"""
Few-Shot Learning

Algorithms that enable models to learn from a small number of training
examples. Includes prototypical networks (Prototypical Networks for
Few-shot Learning, Snell et al., 2017).

https://arxiv.org/abs/1703.05175
"""

from __future__ import annotations

import math
import random


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class PrototypicalNetwork:
    def __init__(self, dim: int):
        self.dim = dim
        self.prototypes: dict[str, list[float]] = {}

    def fit(self, support_set: dict[str, list[list[float]]]) -> None:
        for class_label, examples in support_set.items():
            proto = [0.0] * self.dim
            for ex in examples:
                for i in range(self.dim):
                    proto[i] += ex[i]
            n = len(examples)
            for i in range(self.dim):
                proto[i] /= n
            self.prototypes[class_label] = proto

    def predict(self, query: list[float]) -> str:
        best_label = min(
            self.prototypes,
            key=lambda label: euclidean_distance(query, self.prototypes[label]),
        )
        return best_label


class MatchingNetwork:
    def __init__(self):
        self.support_x: list[list[float]] = []
        self.support_y: list[str] = []

    def fit(self, support_set: dict[str, list[list[float]]]) -> None:
        self.support_x = []
        self.support_y = []
        for label, examples in support_set.items():
            for ex in examples:
                self.support_x.append(ex)
                self.support_y.append(label)

    def predict(self, query: list[float]) -> str:
        similarities = [
            1.0 / (1.0 + euclidean_distance(query, x))
            for x in self.support_x
        ]
        total_sim = sum(similarities) or 1.0
        weights = [s / total_sim for s in similarities]
        class_scores: dict[str, float] = {}
        for w, label in zip(weights, self.support_y):
            class_scores[label] = class_scores.get(label, 0.0) + w
        return max(class_scores, key=class_scores.get)


if __name__ == "__main__":
    random.seed(42)
    dim = 4
    support = {
        "cat": [[random.random() for _ in range(dim)] for _ in range(3)],
        "dog": [[random.random() for _ in range(dim)] for _ in range(3)],
    }
    query_cat = [random.random() for _ in range(dim)]
    query_dog = [random.random() for _ in range(dim)]

    proto = PrototypicalNetwork(dim)
    proto.fit(support)
    print("Prototypical Network:")
    print(f"  Query (cat-like) -> {proto.predict(query_cat)}")
    print(f"  Query (dog-like) -> {proto.predict(query_dog)}")

    match = MatchingNetwork()
    match.fit(support)
    print("\nMatching Network:")
    print(f"  Query (cat-like) -> {match.predict(query_cat)}")
    print(f"  Query (dog-like) -> {match.predict(query_dog)}")
    print("Few-shot learning demo complete.")
