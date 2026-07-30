# Convolutional Neural Network From Scratch

A CNN implemented in pure Python with NumPy (no deep learning frameworks).

Implements a Conv3x3 → MaxPool2 → Softmax pipeline trained with backpropagation on MNIST handwritten digit classification.

## Usage

```bash
pip install -r requirements.txt
python3 cnn.py
```

## Architecture

| Layer | Details | Output Shape |
|-------|---------|--------------|
| Input | 28x28 grayscale image | (28, 28) |
| Conv3x3 | 8 filters, stride 1, valid padding | (26, 26, 8) |
| MaxPool2 | 2x2 non-overlapping windows | (13, 13, 8) |
| Softmax | Fully-connected → 10-class softmax | (10,) |

Based on Victor Zhou's [blog series](https://victorzhou.com/blog/intro-to-cnns-part-1/).
