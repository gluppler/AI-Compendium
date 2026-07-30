# Neural Network From Scratch

A fully commented, modular neural network built from scratch using only NumPy.
No ML frameworks — just math and code.

## What This Teaches

- Forward propagation (how data flows through a network)
- Activation functions and why they matter
- Backpropagation via the chain rule (how the network learns)
- Gradient descent (how weights are updated)
- How architecture choices (width, depth, activations) affect training

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 network.py
```

This trains three different network architectures on a tiny gender-classification
dataset and prints predictions for two unseen people.

## The Three Demos

| Demo | Architecture | Hidden Activation | What It Shows |
|---|---|---|---|
| 1 | 2 - 2 - 1 | sigmoid | Original toy network |
| 2 | 2 - 4 - 1 | sigmoid | Wider layer = more capacity |
| 3 | 2 - 8 - 4 - 1 | ReLU | Deeper + different activation |

## Dataset

| Person | Weight | Height | Gender |
|---|---|---|---|
| Alice | -2 | -1 | Female (1) |
| Bob | 25 | 6 | Male (0) |
| Charlie | 17 | 4 | Male (0) |
| Diana | -15 | -6 | Female (1) |

(Normalized for educational convenience — not real measurements.)

Predictions are made for **Emily** `[-7, -3]` and **Frank** `[20, 2]`.

## Key Concepts Explained in the Code

Every function has inline comments explaining:
- **What** it does
- **Why** it works (the math or intuition)
- **How** it fits into the larger algorithm

## Requirements

- Python 3.7+
- NumPy

## Original

Based on Victor Zhou's [Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/),
rewritten for modularity and deeper educational commentary.
