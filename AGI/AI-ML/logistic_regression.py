import math

def derivative(params, data_points):
    num_features = len(params)
    gradients = [0.0] * num_features
    for features, y_i in data_points:
        z = params[0] + sum(p * x for p, x in zip(params[1:], features))
        prediction = 1.0 / (1.0 + math.exp(-z))
        gradients[0] += prediction - y_i
        for i, x_i in enumerate(features):
            gradients[i + 1] += (prediction - y_i) * x_i
    return gradients

def logistic_regression(data_points, iterations, learning_rate):
    if not data_points:
        return None
    num_features = len(data_points[0][0]) + 1
    params = [0.0] * num_features

    for _ in range(iterations):
        grad = derivative(params, data_points)
        for i in range(len(params)):
            params[i] -= learning_rate * grad[i]

    return params


if __name__ == "__main__":
    data = [([0.0], 0.0), ([1.0], 0.0), ([2.0], 0.0),
            ([3.0], 1.0), ([4.0], 1.0), ([5.0], 1.0)]
    result = logistic_regression(data, 10000, 0.05)
    assert result is not None
    assert abs(result[0] + 17.65) < 1.0
    assert abs(result[1] - 7.13) < 1.0
    assert logistic_regression([], 5000, 0.1) is None
    print("logistic_regression: OK")
