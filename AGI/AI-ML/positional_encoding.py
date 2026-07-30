"""
Positional Encoding

Adds information about token positions in a sequence, since
self-attention is permutation-invariant. Implements the sinusoidal
encoding from "Attention Is All You Need" (Vaswani et al., 2017).

https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import math


def sinusoidal_positional_encoding(
    seq_len: int, d_model: int
) -> list[list[float]]:
    pe = [[0.0] * d_model for _ in range(seq_len)]
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                pe[pos][i] = math.sin(pos / (10000 ** (i / d_model)))
            else:
                pe[pos][i] = math.cos(pos / (10000 ** ((i - 1) / d_model)))
    return pe


class LearnedPositionalEncoding:
    def __init__(self, max_len: int, d_model: int):
        import random
        self.pe = [
            [random.uniform(-0.1, 0.1) for _ in range(d_model)]
            for _ in range(max_len)
        ]

    def forward(self, x: list[list[float]], positions: list[int]) -> list[list[float]]:
        n, d = len(x), len(x[0])
        output = [[0.0] * d for _ in range(n)]
        for i in range(n):
            pos = positions[i] if positions else i
            for j in range(d):
                output[i][j] = x[i][j] + self.pe[pos][j]
        return output


class RelativePositionalBias:
    def __init__(self, max_distance: int, num_heads: int):
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.bias = [
            [0.0 for _ in range(2 * max_distance + 1)]
            for _ in range(num_heads)
        ]

    def get_bias(self, head: int, query_pos: int, key_pos: int) -> float:
        distance = key_pos - query_pos
        clipped = max(-self.max_distance, min(self.max_distance, distance))
        index = clipped + self.max_distance
        return self.bias[head][index]


if __name__ == "__main__":
    seq_len, d_model = 10, 8
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    print("Sinusoidal positional encodings (first 3 positions):")
    for pos in range(3):
        print(f"  pos {pos}: {[round(v, 3) for v in pe[pos][:6]]}...")

    lpe = LearnedPositionalEncoding(seq_len, d_model)
    x = [[0.5 for _ in range(d_model)] for _ in range(3)]
    out = lpe.forward(x, [0, 1, 2])
    print("\nLearned PE output (first position):")
    print(f"  {[round(v, 3) for v in out[0][:6]]}...")

    rpb = RelativePositionalBias(4, 2)
    print(f"\nRelative bias (head=0, q=2, k=5): {rpb.get_bias(0, 2, 5):.3f}")
    print(f"Relative bias (head=0, q=2, k=0): {rpb.get_bias(0, 2, 0):.3f}")
    print("Positional encoding demo complete.")
