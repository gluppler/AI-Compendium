# Recurrent Neural Network — Sentiment Classifier from Scratch

A many-to-one Vanilla Recurrent Neural Network implemented from scratch using only NumPy. The RNN is trained to classify short English phrases as **positive** or **negative** sentiment.

Built on the concepts from Victor Zhou's [Introduction to Recurrent Neural Networks](https://victorzhou.com/blog/intro-to-rnns/).

## How It Works

### Architecture
- **Input**: One-hot encoded vectors representing each word in a phrase
- **Hidden layer**: 64 neurons with `tanh` activation
- **Output**: 2 neurons (positive / negative), passed through softmax
- **Training**: Backpropagation Through Time (BPTT) with gradient clipping

### Forward Pass
For each word in the input, the RNN updates a hidden state `h`:
```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
```
After processing all words, the final hidden state is used to produce the output:
```
y = W_hy @ h_n + b_y
```

### Training
- Loss function: Categorical cross-entropy
- Optimizer: Stochastic Gradient Descent (learning rate = 0.02)
- Gradient clipping to [-1, 1] to prevent exploding gradients

## Usage

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train the model

```bash
python rnn_sentiment.py
```

Trains for 1000 epochs, printing train/test loss and accuracy every 100 epochs.

### Train with custom parameters

```bash
python rnn_sentiment.py --epochs 2000 --hidden-size 128 --learning-rate 0.01
```

### Save model weights

```bash
python rnn_sentiment.py --save model.npz
```

### Load weights and predict a single phrase

```bash
python rnn_sentiment.py --load model.npz --predict "this is very good"
```

### Interactive prediction mode

```bash
python rnn_sentiment.py --load model.npz --predict
```

Then type phrases at the `>` prompt.

### Plot training loss

```bash
python rnn_sentiment.py --plot
```

## Sample Results

After 1000 epochs with default settings:

| Set     | Loss  | Accuracy |
|---------|-------|----------|
| Train   | ~0.05 | ~0.97    |
| Test    | ~0.40 | ~0.85    |

## Project Structure

```
Recurrent_Neural_Network/
├── rnn_sentiment.py   # Main script — RNN class, data, training, prediction
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── venv/              # Virtual environment (optional)
```

## Dependencies

- numpy >= 1.21.0
- matplotlib >= 3.4.0 (optional — only needed for `--plot`)

## License

MIT
