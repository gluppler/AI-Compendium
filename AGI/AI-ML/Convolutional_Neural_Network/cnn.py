import numpy as np


class Conv3x3:
    def __init__(self, num_filters: int) -> None:
        self.num_filters = num_filters
        self.filters: np.ndarray = np.random.randn(num_filters, 3, 3) / 9
        self.last_input: np.ndarray | None = None

    def _iterate_regions(self, image: np.ndarray):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                yield image[i:i + 3, j:j + 3], i, j

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))
        for im_region, i, j in self._iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backprop(self, d_L_d_out: np.ndarray, learn_rate: float) -> np.ndarray:
        assert self.last_input is not None
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self._iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
                d_L_d_input[i:i + 3, j:j + 3] += d_L_d_out[i, j, f] * self.filters[f]

        self.filters -= learn_rate * d_L_d_filters
        return d_L_d_input


class MaxPool2:
    def _iterate_regions(self, image: np.ndarray):
        h, w, _ = image.shape
        for i in range(h // 2):
            for j in range(w // 2):
                yield image[i * 2:i * 2 + 2, j * 2:j * 2 + 2], i, j

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))
        for im_region, i, j in self._iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backprop(self, d_L_d_out: np.ndarray) -> np.ndarray:
        d_L_d_input = np.zeros(self.last_input.shape)
        for im_region, i, j in self._iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]
        return d_L_d_input


class Softmax:
    def __init__(self, input_len: int, nodes: int) -> None:
        self.weights: np.ndarray = np.random.randn(input_len, nodes) / input_len
        self.biases: np.ndarray = np.zeros(nodes)
        self.last_input_shape: tuple[int, ...] = ()
        self.last_input: np.ndarray | None = None
        self.last_totals: np.ndarray | None = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.last_input_shape = input.shape
        flat = input.flatten()
        self.last_input = flat
        totals = flat @ self.weights + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp)

    def backprop(self, d_L_d_out: np.ndarray, learn_rate: float) -> np.ndarray:
        assert self.last_totals is not None
        assert self.last_input is not None
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
            d_L_d_t = gradient * d_out_d_t
            d_L_d_w = self.last_input[:, np.newaxis] @ d_L_d_t[np.newaxis, :]
            d_L_d_b = d_L_d_t
            d_L_d_inputs = self.weights @ d_L_d_t
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)
        return np.zeros(self.last_input_shape)


def forward(image: np.ndarray, label: int,
            conv: Conv3x3, pool: MaxPool2, softmax: Softmax
            ) -> tuple[np.ndarray, float, int]:
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out, loss, acc


def train(image: np.ndarray, label: int,
          conv: Conv3x3, pool: MaxPool2, softmax: Softmax,
          lr: float = 0.005
          ) -> tuple[float, int]:
    out, loss, acc = forward(image, label, conv, pool, softmax)
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)
    return loss, acc


def main() -> None:
    import mnist
    mnist.datasets_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

    train_images = mnist.train_images()[:1000]
    train_labels = mnist.train_labels()[:1000]
    test_images = mnist.test_images()[:1000]
    test_labels = mnist.test_labels()[:1000]

    conv = Conv3x3(8)
    pool = MaxPool2()
    softmax = Softmax(13 * 13 * 8, 10)

    print("MNIST CNN initialized!\n")

    for epoch in range(3):
        print(f"--- Epoch {epoch + 1} ---")
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]
        loss = 0.0
        num_correct = 0

        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            step_loss, acc = train(im, label, conv, pool, softmax)
            loss += step_loss
            num_correct += acc
            if i % 100 == 99:
                print(f"[Step {i + 1}] Past 100 steps: Average Loss {loss / 100:.3f} | Accuracy: {num_correct}%")
                loss = 0.0
                num_correct = 0

    print("\n--- Testing the CNN ---")
    loss = 0.0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        _, step_loss, acc = forward(im, label, conv, pool, softmax)
        loss += step_loss
        num_correct += acc
    num_tests = len(test_images)
    print(f"Test Loss: {loss / num_tests:.3f}")
    print(f"Test Accuracy: {num_correct / num_tests:.3f}")


if __name__ == "__main__":
    main()
