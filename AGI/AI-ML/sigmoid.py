import math

def sigmoid(array):
    for i in range(len(array)):
        array[i] = 1.0 / (1.0 + math.exp(-array[i]))
    return array


if __name__ == "__main__":
    test = [1.0, 0.5, -1.0, 0.0, 0.3]
    result = sigmoid(test[:])
    expected = [0.7310586, 0.62245935, 0.26894143, 0.5, 0.5744425]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-5, f"{r} != {e}"
    print("sigmoid: OK")
