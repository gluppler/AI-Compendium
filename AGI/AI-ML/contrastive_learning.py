"""
Contrastive Learning

Self-supervised representation learning that pulls similar samples
together and pushes dissimilar samples apart. Implements the NT-Xent
(Normalized Temperature-Scaled Cross-Entropy) loss used in SimCLR.

https://arxiv.org/abs/2002.05709
"""

from __future__ import annotations

import math
import random


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) + 1e-10
    norm_b = math.sqrt(sum(y * y for y in b)) + 1e-10
    return dot / (norm_a * norm_b)


def nt_xent_loss(
    embeddings: list[list[float]],
    temperature: float = 0.5,
) -> tuple[float, int]:
    n = len(embeddings)
    total_loss = 0.0
    num_pairs = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sim_ij = cosine_similarity(embeddings[i], embeddings[j]) / temperature
            is_positive = (i % 2 == 0 and j == i + 1) or (j % 2 == 0 and i == j + 1)
            denom = 0.0
            for k in range(n):
                if k == i:
                    continue
                denom += math.exp(cosine_similarity(embeddings[i], embeddings[k]) / temperature)
            if is_positive:
                loss = -math.log(math.exp(sim_ij) / denom)
                total_loss += loss
                num_pairs += 1
    return total_loss / max(num_pairs, 1), num_pairs


class SimCLRProjection:
    def __init__(self, input_dim: int, projection_dim: int):
        self.w = [
            [random.uniform(-0.1, 0.1) for _ in range(projection_dim)]
            for _ in range(input_dim)
        ]
        self.b = [0.0] * projection_dim

    def project(self, x: list[float]) -> list[float]:
        return [
            sum(x[j] * self.w[j][i] for j in range(len(x))) + self.b[i]
            for i in range(len(self.b))
        ]


if __name__ == "__main__":
    random.seed(42)
    projector = SimCLRProjection(8, 4)
    original = [[random.random() for _ in range(8)] for _ in range(3)]
    augmented = [[v + random.gauss(0, 0.05) for v in orig] for orig in original]
    batch = []
    for orig, aug in zip(original, augmented):
        batch.append(projector.project(orig))
        batch.append(projector.project(aug))

    loss, count = nt_xent_loss(batch)
    print(f"NT-Xent loss: {loss:.4f} over {count} positive pairs")
    print(f"Positive pairs (close): {count}")

    sim_same_1 = cosine_similarity(batch[0], batch[1])
    sim_diff = cosine_similarity(batch[0], batch[2])
    print(f"Similarity (same image, 2 views): {sim_same_1:.3f}")
    print(f"Similarity (different images):    {sim_diff:.3f}")
    print("Contrastive learning demo complete.")
