"""
Transformer Block

Encoder and decoder blocks from "Attention Is All You Need" (Vaswani et al., 2017).
Each block contains multi-head self-attention, feed-forward network, residual
connections, and layer normalization.

https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

from ai_ml.attention_mechanisms import MultiHeadAttention


def layer_norm(x: list[list[float]], eps: float = 1e-5) -> list[list[float]]:
    n_rows, n_cols = len(x), len(x[0])
    output = [[0.0] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        mean = sum(x[i]) / n_cols
        var = sum((v - mean) ** 2 for v in x[i]) / n_cols
        for j in range(n_cols):
            output[i][j] = (x[i][j] - mean) / ((var + eps) ** 0.5)
    return output


def feed_forward(
    x: list[list[float]],
    w1: list[list[float]],
    b1: list[float],
    w2: list[list[float]],
    b2: list[float],
) -> list[list[float]]:
    n, d_ff = len(x), len(w1[0])
    hidden = [[0.0] * d_ff for _ in range(n)]
    for i in range(n):
        for j in range(d_ff):
            hidden[i][j] = sum(x[i][k] * w1[k][j] for k in range(len(x[0]))) + b1[j]
            hidden[i][j] = max(0.0, hidden[i][j])

    output = [[0.0] * len(b2) for _ in range(n)]
    for i in range(n):
        for j in range(len(b2)):
            output[i][j] = sum(hidden[i][k] * w2[k][j] for k in range(d_ff)) + b2[j]
    return output


class EncoderBlock:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        attn_out = self.attention.forward(x, x, x)
        x = [[x[i][j] + attn_out[i][j] for j in range(self.d_model)] for i in range(len(x))]
        x = layer_norm(x)

        d_ff_out = feed_forward(
            x,
            [[0.02 for _ in range(self.d_ff)] for _ in range(self.d_model)],
            [0.0] * self.d_ff,
            [[0.02 for _ in range(self.d_model)] for _ in range(self.d_ff)],
            [0.0] * self.d_model,
        )
        x = [[x[i][j] + d_ff_out[i][j] for j in range(self.d_model)] for i in range(len(x))]
        x = layer_norm(x)
        return x


if __name__ == "__main__":
    d_model, num_heads, d_ff = 8, 2, 16
    seq = [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(4)]
    block = EncoderBlock(d_model, num_heads, d_ff)
    output = block.forward(seq)
    print("Encoder block output:")
    for row in output:
        print(f"  {[round(v, 3) for v in row]}")
    print(f"Shape: {len(output)} x {len(output[0])}")
    print("Transformer block demo complete.")
