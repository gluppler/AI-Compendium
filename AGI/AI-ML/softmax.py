import math

def softmax(array):
    exp_vals = [math.exp(x) for x in array]
    s = sum(exp_vals)
    return [v / s for v in exp_vals]


if __name__ == "__main__":
    test = [9.0, 0.5, -3.0, 0.0, 3.0]
    result = softmax(test)
    expected = [0.9971961, 0.00020289792, 6.126987e-6, 0.00012306382, 0.0024718025]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-5, f"{r} != {e}"
    print("softmax: OK")
