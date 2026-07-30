import random
import math
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self):
        self.trees = []
        self.feature_indices = []
        self.num_classes = 0

    def fit(self, training_data, num_trees=10, max_depth=5,
            min_samples_split=2, max_features=None):
        if not training_data:
            return None
        num_features = len(training_data[0][0])
        if num_features == 0:
            return None

        if max_features is None:
            max_features = max(1, int(math.sqrt(num_features)))
        max_features = min(max_features, num_features)

        self.trees = []
        self.feature_indices = []

        for _ in range(num_trees):
            feature_idxs = random.sample(range(num_features), max_features)

            bootstrap = []
            for _ in range(len(training_data)):
                bootstrap.append(random.choice(training_data))

            limited_sample = []
            for features, label in bootstrap:
                limited = [features[i] for i in feature_idxs]
                limited_sample.append((limited, label))

            tree = DecisionTree().fit(limited_sample, max_depth, min_samples_split)
            if tree is not None:
                self.trees.append(tree)
                self.feature_indices.append(feature_idxs)

        if not self.trees:
            return None

        unique_labels = set(l for _, l in training_data)
        self.num_classes = len(unique_labels)
        return self

    def predict(self, test_point):
        if not test_point or not self.trees:
            return None

        predictions = []
        for tree, feat_idxs in zip(self.trees, self.feature_indices):
            limited = [test_point[i] for i in feat_idxs]
            pred = tree.predict(limited)
            if pred is not None:
                predictions.append(pred)

        if not predictions:
            return None

        counts = {}
        for p in predictions:
            counts[p] = counts.get(p, 0) + 1
        return max(counts, key=counts.get)

    def predict_batch(self, test_points):
        return [self.predict(p) for p in test_points]


def random_forest(training_data, test_point, num_trees, max_depth,
                  min_samples_split, max_features):
    model = RandomForest().fit(training_data, num_trees, max_depth,
                                min_samples_split, max_features)
    if model is None:
        return None
    return model.predict(test_point)


if __name__ == "__main__":
    data = [
        ([1.0, 1.0], 0.0), ([2.0, 2.0], 0.0), ([3.0, 3.0], 0.0),
        ([5.0, 5.0], 1.0), ([6.0, 6.0], 1.0), ([7.0, 7.0], 1.0),
    ]
    result = random_forest(data, [1.5, 1.5], 10, 5, 2, None)
    assert result == 0.0, f"got {result}"
    result = random_forest(data, [5.5, 5.5], 10, 5, 2, None)
    assert result == 1.0, f"got {result}"
    assert random_forest([], [1.0], 10, 5, 2, None) is None
    print("random_forest: OK")
