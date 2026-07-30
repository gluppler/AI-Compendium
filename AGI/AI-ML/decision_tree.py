import math

class TreeNode:
    def __init__(self, class_label=None, feature_index=None, threshold=None,
                 left=None, right=None, samples=0):
        self.class_label = class_label
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.samples = samples
        self.is_leaf = class_label is not None

def calculate_entropy(labels):
    if not labels:
        return 0.0
    total = len(labels)
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            entropy -= p * math.log2(p)
    return entropy

def get_majority_class(labels):
    if not labels:
        return 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return max(counts, key=counts.get)

def find_best_split(data, feature_index):
    if not data:
        return None
    num_samples = len(data)
    feature_values = [(features[feature_index], label) for features, label in data]
    feature_values.sort(key=lambda x: x[0])

    parent_entropy = calculate_entropy([l for _, l in data])

    best_threshold = feature_values[0][0]
    best_gain = 0.0

    for i in range(1, num_samples):
        if feature_values[i][0] != feature_values[i - 1][0]:
            threshold = (feature_values[i][0] + feature_values[i - 1][0]) / 2.0

            left_labels = [l for _, l in feature_values[:i]]
            right_labels = [l for _, l in feature_values[i:]]

            left_entropy = calculate_entropy(left_labels)
            right_entropy = calculate_entropy(right_labels)

            left_weight = i / num_samples
            right_weight = (num_samples - i) / num_samples

            weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
            information_gain = parent_entropy - weighted_entropy

            if information_gain > best_gain:
                best_gain = information_gain
                best_threshold = threshold

    return (best_threshold, best_gain) if best_gain > 0.0 else None

def find_best_split_feature(data, feature_indices):
    if not data or not feature_indices:
        return None
    best_feature_index = 0
    best_threshold = 0.0
    best_gain = 0.0

    for fi in feature_indices:
        result = find_best_split(data, fi)
        if result is not None:
            threshold, gain = result
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                best_feature_index = fi

    return (best_feature_index, best_threshold) if best_gain > 0.0 else None

def build_tree(data, feature_indices, max_depth, min_samples_split, current_depth):
    labels = [l for _, l in data]
    unique_labels = set(labels)

    if (len(unique_labels) == 1 or len(data) < min_samples_split
            or current_depth >= max_depth or not feature_indices):
        return TreeNode(class_label=get_majority_class(labels), samples=len(data))

    result = find_best_split_feature(data, feature_indices)
    if result is None:
        return TreeNode(class_label=get_majority_class(labels), samples=len(data))

    feature_index, threshold = result
    left_data = [(f, l) for f, l in data if f[feature_index] < threshold]
    right_data = [(f, l) for f, l in data if f[feature_index] >= threshold]

    if not left_data or not right_data:
        return TreeNode(class_label=get_majority_class(labels), samples=len(data))

    left_child = build_tree(left_data, feature_indices, max_depth,
                            min_samples_split, current_depth + 1)
    right_child = build_tree(right_data, feature_indices, max_depth,
                             min_samples_split, current_depth + 1)

    return TreeNode(feature_index=feature_index, threshold=threshold,
                    left=left_child, right=right_child, samples=len(data))

def predict_tree(tree, features):
    if tree.is_leaf:
        return tree.class_label
    if features[tree.feature_index] < tree.threshold:
        return predict_tree(tree.left, features)
    else:
        return predict_tree(tree.right, features)

class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, training_data, max_depth=10, min_samples_split=2):
        if not training_data:
            return None
        num_features = len(training_data[0][0])
        if num_features == 0:
            return None

        feature_indices = list(range(num_features))
        self.tree = build_tree(training_data, feature_indices, max_depth,
                               min_samples_split, 0)
        return self

    def predict(self, test_point):
        if not test_point or self.tree is None:
            return None
        return predict_tree(self.tree, test_point)

    def predict_batch(self, test_points):
        return [self.predict(p) for p in test_points]

def decision_tree(training_data, test_point, max_depth, min_samples_split):
    model = DecisionTree().fit(training_data, max_depth, min_samples_split)
    if model is None:
        return None
    return model.predict(test_point)


if __name__ == "__main__":
    data = [
        ([1.0, 1.0], 0.0), ([2.0, 2.0], 0.0), ([3.0, 3.0], 0.0),
        ([5.0, 5.0], 1.0), ([6.0, 6.0], 1.0), ([7.0, 7.0], 1.0),
    ]
    assert decision_tree(data, [1.5, 1.5], 10, 2) == 0.0
    assert decision_tree(data, [5.5, 5.5], 10, 2) == 1.0
    assert decision_tree([], [1.0], 10, 2) is None

    ent = calculate_entropy([0.0, 0.0, 0.0, 0.0])
    assert abs(ent) < 1e-10
    ent = calculate_entropy([0.0, 0.0, 1.0, 1.0])
    assert abs(ent - 1.0) < 1e-10
    print("decision_tree: OK")
