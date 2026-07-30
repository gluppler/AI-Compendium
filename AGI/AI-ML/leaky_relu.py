def leaky_relu(vector, alpha):
    return [x if x >= 0 else alpha * x for x in vector]


if __name__ == "__main__":
    test = [-10.0, 2.0, -3.0, 4.0, -5.0, 10.0, 0.05]
    result = leaky_relu(test, 0.01)
    assert result == [-0.1, 2.0, -0.03, 4.0, -0.05, 10.0, 0.05]
    print("leaky_relu: OK")
