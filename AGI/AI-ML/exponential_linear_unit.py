import math

def exponential_linear_unit(vector, alpha):
    result = []
    for x in vector:
        if x >= 0:
            result.append(x)
        else:
            result.append(x * alpha * (math.exp(x) - 1.0))
    return result


if __name__ == "__main__":
    test = [-10.0, 2.0, -3.0, 4.0, -5.0, 10.0, 0.05]
    result = exponential_linear_unit(test, 0.01)
    expected = [0.09999546000702375, 2.0, 0.028506387948964082, 4.0, 0.049663102650045726, 10.0, 0.05]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-5, f"{r} != {e}"
    print("elu: OK")
