"""
Attention Mechanisms

Implementations of scaled dot-product attention and multi-head attention
as described in "Attention Is All You Need" (Vaswani et al., 2017).

https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import math


def scaled_dot_product_attention(
    query: list[list[float]],
    key: list[list[float]],
    value: list[list[float]],
    mask: list[list[float]] | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    n = len(query)
    d_k = len(query[0])
    scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            scores[i][j] = sum(query[i][k] * key[j][k] for k in range(d_k))
    scale = math.sqrt(d_k)
    for i in range(n):
        for j in range(n):
            scores[i][j] /= scale

    if mask:
        for i in range(n):
            for j in range(n):
                if mask[i][j] == -float("inf"):
                    scores[i][j] = -float("inf")

    for i in range(n):
        max_val = max(scores[i])
        exp_sum = 0.0
        for j in range(n):
            scores[i][j] = math.exp(scores[i][j] - max_val)
            exp_sum += scores[i][j]
        for j in range(n):
            scores[i][j] /= exp_sum

    output = [[0.0] * len(value[0]) for _ in range(n)]
    for i in range(n):
        for j in range(len(value[0])):
            output[i][j] = sum(scores[i][k] * value[k][j] for k in range(n))

    return output, scores


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def _split_heads(self, x: list[list[float]]) -> list[list[list[float]]]:
        n = len(x)
        heads: list[list[list[float]]] = []
        for h in range(self.num_heads):
            start = h * self.d_k
            end = start + self.d_k
            head = [[row[c] for c in range(start, end)] for row in x]
            heads.append(head)
        return heads

    def _combine_heads(self, heads: list[list[list[float]]]) -> list[list[float]]:
        n = len(heads[0])
        output = [[0.0] * self.d_model for _ in range(n)]
        for i in range(n):
            col = 0
            for h in range(self.num_heads):
                for k in range(self.d_k):
                    output[i][col] = heads[h][i][k]
                    col += 1
        return output

    def forward(
        self,
        query: list[list[float]],
        key: list[list[float]],
        value: list[list[float]],
    ) -> list[list[float]]:
        q_heads = self._split_heads(query)
        k_heads = self._split_heads(key)
        v_heads = self._split_heads(value)

        attended_heads = []
        for h in range(self.num_heads):
            attn_out, _ = scaled_dot_product_attention(
                q_heads[h], k_heads[h], v_heads[h]
            )
            attended_heads.append(attn_out)

        return self._combine_heads(attended_heads)


if __name__ == "__main__":
    d_model = 8
    seq = [
        [0.1 * (i + j + 1) for j in range(d_model)]
        for i in range(4)
    ]
    mha = MultiHeadAttention(d_model, num_heads=2)
    result = mha.forward(seq, seq, seq)
    print("Multi-head attention output:")
    for row in result:
        print(f"  {[round(v, 3) for v in row]}")
    print(f"Shape: {len(result)} x {len(result[0])}")
    print("Attention mechanisms demo complete.")
