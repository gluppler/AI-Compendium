import math

def tanh(array):
    for i in range(len(array)):
        array[i] = (2.0 / (1.0 + math.exp(-2.0 * array[i]))) - 1.0
    return array


if __name__ == "__main__":
    test = [1.0, 0.5, -1.0, 0.0, 0.3]
    result = tanh(test[:])
    expected = [0.76159406, 0.4621172, -0.7615941, 0.0, 0.29131258]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-5, f"{r} != {e}"
    print("tanh: OK")
