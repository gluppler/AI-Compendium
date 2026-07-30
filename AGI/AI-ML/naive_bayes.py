import math

class ClassStatistics:
    def __init__(self, class_label, prior, feature_means, feature_variances):
        self.class_label = class_label
        self.prior = prior
        self.feature_means = feature_means
        self.feature_variances = feature_variances

def gaussian_log_pdf(x, mean, variance):
    diff = x - mean
    exponent = -(diff * diff) / (2.0 * variance)
    log_coefficient = -0.5 * math.log(2.0 * math.pi * variance)
    return log_coefficient + exponent

def calculate_class_statistics(training_data, class_label, num_features):
    class_samples = [(f, l) for f, l in training_data if abs(l - class_label) < 1e-10]
    if not class_samples:
        return None

    prior = len(class_samples) / len(training_data)

    feature_means = [0.0] * num_features
    for features, _ in class_samples:
        for i in range(num_features):
            feature_means[i] += features[i]
    n = len(class_samples)
    for i in range(num_features):
        feature_means[i] /= n

    feature_variances = [0.0] * num_features
    for features, _ in class_samples:
        for i in range(num_features):
            diff = features[i] - feature_means[i]
            feature_variances[i] += diff * diff
    for i in range(num_features):
        feature_variances[i] = max(feature_variances[i] / n, 1e-9)

    return ClassStatistics(class_label, prior, feature_means, feature_variances)

def train_naive_bayes(training_data):
    if not training_data:
        return None
    num_features = len(training_data[0][0])
    if num_features == 0:
        return None
    if not all(len(f) == num_features for f, _ in training_data):
        return None

    unique_classes = []
    for _, label in training_data:
        if not any(abs(c - label) < 1e-10 for c in unique_classes):
            unique_classes.append(label)

    class_stats = []
    for label in unique_classes:
        stats = calculate_class_statistics(training_data, label, num_features)
        if stats:
            class_stats.append(stats)

    return class_stats if class_stats else None

def predict_naive_bayes(model, test_point):
    if not model or not test_point:
        return None
    if len(test_point) != len(model[0].feature_means):
        return None

    best_class = None
    best_log_prob = -float('inf')

    for stats in model:
        log_prob = math.log(stats.prior)
        for i, feature in enumerate(test_point):
            if i < len(stats.feature_means) and i < len(stats.feature_variances):
                log_prob += gaussian_log_pdf(feature, stats.feature_means[i],
                                              stats.feature_variances[i])
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_class = stats.class_label

    return best_class

def naive_bayes(training_data, test_point):
    model = train_naive_bayes(training_data)
    if model is None:
        return None
    return predict_naive_bayes(model, test_point)


if __name__ == "__main__":
    data = [
        ([1.0, 1.0], 0.0),
        ([1.1, 1.0], 0.0),
        ([1.0, 1.1], 0.0),
        ([5.0, 5.0], 1.0),
        ([5.1, 5.0], 1.0),
        ([5.0, 5.1], 1.0),
    ]
    assert naive_bayes(data, [1.05, 1.05]) == 0.0
    assert naive_bayes(data, [5.05, 5.05]) == 1.0
    assert naive_bayes([], [1.0, 2.0]) is None
    print("naive_bayes: OK")
