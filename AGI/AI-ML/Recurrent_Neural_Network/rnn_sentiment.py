import numpy as np
import random
import argparse
import sys


class RNN:
    def __init__(self, input_size, output_size, hidden_size=64):
        self.Whh = np.random.randn(hidden_size, hidden_size) / 1000
        self.Wxh = np.random.randn(hidden_size, input_size) / 1000
        self.Why = np.random.randn(output_size, hidden_size) / 1000
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        self.last_inputs = inputs
        self.last_hs = {0: h}
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h
        y = self.Why @ h + self.by
        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)
        d_h = self.Why.T @ d_y
        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
            d_bh += temp
            d_Whh += temp @ self.last_hs[t].T
            d_Wxh += temp @ self.last_inputs[t].T
            d_h = self.Whh @ temp
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


TRAIN_DATA = {
    'good': True,
    'bad': False,
    'happy': True,
    'sad': False,
    'not good': False,
    'not bad': True,
    'not happy': False,
    'not sad': True,
    'very good': True,
    'very bad': False,
    'very happy': True,
    'very sad': False,
    'i am happy': True,
    'this is good': True,
    'i am bad': False,
    'this is bad': False,
    'i am sad': False,
    'this is sad': False,
    'i am not happy': False,
    'this is not good': False,
    'i am not bad': True,
    'this is not sad': True,
    'i am very happy': True,
    'this is very good': True,
    'i am very bad': False,
    'this is very sad': False,
    'this is very happy': True,
    'i am good not bad': True,
    'this is good not bad': True,
    'i am bad not good': False,
    'i am good and happy': True,
    'this is not good and not happy': False,
    'i am not at all good': False,
    'i am not at all bad': True,
    'i am not at all happy': False,
    'this is not at all sad': True,
    'this is not at all happy': False,
    'i am good right now': True,
    'i am bad right now': False,
    'this is bad right now': False,
    'i am sad right now': False,
    'i was good earlier': True,
    'i was happy earlier': True,
    'i was bad earlier': False,
    'i was sad earlier': False,
    'i am very bad right now': False,
    'this is very good right now': True,
    'this is very sad right now': False,
    'this was bad earlier': False,
    'this was very good earlier': True,
    'this was very bad earlier': False,
    'this was very happy earlier': True,
    'this was very sad earlier': False,
    'i was good and not bad earlier': True,
    'i was not good and not happy earlier': False,
    'i am not at all bad or sad right now': True,
    'i am not at all good or happy right now': False,
    'this was not happy and not good earlier': False,
}

TEST_DATA = {
    'this is happy': True,
    'i am good': True,
    'this is not happy': False,
    'i am not good': False,
    'this is not bad': True,
    'i am not sad': True,
    'i am very good': True,
    'this is very bad': False,
    'i am very sad': False,
    'this is bad not good': False,
    'this is good and happy': True,
    'i am not good and not happy': False,
    'i am not at all sad': True,
    'this is not at all good': False,
    'this is not at all bad': True,
    'this is good right now': True,
    'this is sad right now': False,
    'this is very bad right now': False,
    'this was good earlier': True,
    'i was not happy and not good earlier': False,
}


def softmax(xs):
    return np.exp(xs) / sum(np.exp(xs))


def build_vocab(data):
    vocab = sorted(set([w for text in data.keys() for w in text.split(' ')]))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}
    return vocab, word_to_idx, idx_to_word


def create_inputs(text, word_to_idx, vocab_size):
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs


def process_data(rnn, data, word_to_idx, vocab_size, backprop=True, learn_rate=2e-2):
    items = list(data.items())
    random.shuffle(items)
    loss = 0
    num_correct = 0
    for x, y in items:
        inputs = create_inputs(x, word_to_idx, vocab_size)
        target = int(y)
        out, _ = rnn.forward(inputs)
        probs = softmax(out)
        loss -= float(np.log(probs[target, 0]))
        num_correct += int(np.argmax(probs) == target)
        if backprop:
            d_L_d_y = np.copy(probs)
            d_L_d_y[target, 0] -= 1
            rnn.backprop(d_L_d_y, learn_rate)
    return loss / len(data), num_correct / len(data)


def train(args):
    vocab, word_to_idx, idx_to_word = build_vocab(TRAIN_DATA)
    vocab_size = len(vocab)
    print(f'Vocabulary: {vocab_size} unique words\n')

    rnn = RNN(vocab_size, 2, hidden_size=args.hidden_size)
    learn_rate = args.learning_rate

    losses = []
    for epoch in range(args.epochs):
        train_loss, train_acc = process_data(
            rnn, TRAIN_DATA, word_to_idx, vocab_size,
            backprop=True, learn_rate=learn_rate
        )
        losses.append(train_loss)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            test_loss, test_acc = process_data(
                rnn, TEST_DATA, word_to_idx, vocab_size, backprop=False
            )
            print(f'Epoch {epoch + 1:4d} | Train Loss: {train_loss:.3f} | '
                  f'Train Acc: {train_acc:.3f} | Test Loss: {test_loss:.3f} | '
                  f'Test Acc: {test_acc:.3f}')

    if args.plot and args.epochs > 1:
        try:
            import matplotlib.pyplot as plt
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss over Time')
            plt.savefig('training_loss.png')
            print('\nTraining loss plot saved to training_loss.png')
            plt.show()
        except ImportError:
            print('\nmatplotlib not installed; skipping plot.')

    if args.save:
        np.savez(args.save,
                 Whh=rnn.Whh, Wxh=rnn.Wxh, Why=rnn.Why,
                 bh=rnn.bh, by=rnn.by)
        print(f'Model weights saved to {args.save}')

    return rnn, word_to_idx, idx_to_word, vocab_size


def load_weights(path, input_size, output_size, hidden_size=64):
    data = np.load(path)
    rnn = RNN(input_size, output_size, hidden_size)
    rnn.Whh = data['Whh']
    rnn.Wxh = data['Wxh']
    rnn.Why = data['Why']
    rnn.bh = data['bh']
    rnn.by = data['by']
    return rnn


def predict(args, rnn=None, word_to_idx=None, idx_to_word=None, vocab_size=None):
    if rnn is None:
        vocab, word_to_idx, idx_to_word = build_vocab(TRAIN_DATA)
        vocab_size = len(vocab)
        print('Loading model weights...')
        rnn = load_weights(args.load, vocab_size, 2, args.hidden_size)

    if args.predict != 'interactive':
        texts = [args.predict]
    else:
        print('\nEnter text to analyze (or "quit" to exit):')

        def read_input():
            try:
                return input('> ')
            except EOFError:
                return 'quit'
        texts = iter(read_input, 'quit')

    for text in texts:
        if text.lower() == 'quit':
            break
        words = text.strip().split()
        unknown = [w for w in words if w not in word_to_idx]
        if unknown:
            print(f'  Unknown word(s): {unknown}')
            print(f'  Try: {", ".join(word_to_idx.keys())}')
            continue
        inputs = create_inputs(text, word_to_idx, vocab_size)
        out, _ = rnn.forward(inputs)
        probs = softmax(out).flatten()
        pred = 'POSITIVE' if np.argmax(probs) == 1 else 'NEGATIVE'
        confidence = float(probs[int(np.argmax(probs))])
        print(f'  "{text}" -> {pred} (confidence: {confidence:.3f})')


def main():
    parser = argparse.ArgumentParser(description='RNN Sentiment Classifier')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Size of hidden layer (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=2e-2,
                        help='Learning rate (default: 0.02)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save model weights to .npz file')
    parser.add_argument('--load', type=str, default=None,
                        help='Load model weights from .npz file')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training loss curve (requires matplotlib)')
    parser.add_argument('--predict', type=str, default=None, nargs='?',
                        const='interactive', metavar='TEXT',
                        help='Run prediction. If no TEXT given, enters interactive mode.')

    args = parser.parse_args()

    if args.load and args.predict is not None:
        predict(args)
    elif args.predict is not None:
        rnn, word_to_idx, idx_to_word, vocab_size = train(args)
        predict(args, rnn, word_to_idx, idx_to_word, vocab_size)
    else:
        train(args)


if __name__ == '__main__':
    main()
