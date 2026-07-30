import math

def _tanh(x):
    return (2.0 / (1.0 + math.exp(-2.0 * x))) - 1.0

def gaussian_error_linear_unit(vector):
    result = []
    for x in vector:
        gelu = x * 0.5 * (1.0 + _tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
        result.append(gelu)
    return result


if __name__ == "__main__":
    test = [-10.0, 2.0, -3.0, 4.0, -5.0, 10.0, 0.05]
    result = gaussian_error_linear_unit(test)
    expected = [-0.0, 1.9545976940877752, -0.0036373920817729943, 3.9999297540518075, -2.2917961972623857e-7, 10.0, 0.025996938238622008]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-5, f"{r} != {e}"
    print("gelu: OK")
