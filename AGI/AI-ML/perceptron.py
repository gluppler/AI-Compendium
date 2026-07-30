def predict(weights, bias, features):
    s = sum(w * f for w, f in zip(weights, features)) + bias
    return 1.0 if s >= 0.0 else -1.0

def classify(weights, bias, features):
    if not weights or not features:
        return None
    if len(weights) != len(features):
        return None
    return predict(weights, bias, features)

def perceptron(data_points, max_iterations, learning_rate):
    if not data_points:
        return None
    num_features = len(data_points[0][0])
    if num_features == 0:
        return None

    weights = [0.0] * num_features
    bias = 0.0

    for _ in range(max_iterations):
        misclassified = 0
        for features, label in data_points:
            pred = predict(weights, bias, features)
            if pred != label:
                misclassified += 1
                for j in range(len(weights)):
                    weights[j] += learning_rate * label * features[j]
                bias += learning_rate * label
        if misclassified == 0:
            break

    return weights, bias


if __name__ == "__main__":
    data = [
        ([1.0, 1.0], 1.0),
        ([2.0, 2.0], 1.0),
        ([3.0, 3.0], 1.0),
        ([-1.0, -1.0], -1.0),
        ([-2.0, -2.0], -1.0),
        ([-3.0, -3.0], -1.0),
    ]
    result = perceptron(data, 100, 0.1)
    assert result is not None
    w, b = result
    assert classify(w, b, [2.5, 2.5]) == 1.0
    assert classify(w, b, [-2.5, -2.5]) == -1.0
    print("perceptron: OK")
