import math

def euclidean_distance(p1, p2):
    if len(p1) != len(p2):
        return float('inf')
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def k_nearest_neighbors(training_data, test_point, k):
    if not training_data or k == 0 or k > len(training_data):
        return None

    distances = [(euclidean_distance(test_point, features), label)
                 for features, label in training_data]
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    counts = {}
    for _, label in k_nearest:
        counts[label] = counts.get(label, 0) + 1

    return max(counts, key=counts.get)


if __name__ == "__main__":
    data = [
        ([0.0, 0.0], 0.0),
        ([1.0, 0.0], 0.0),
        ([0.0, 1.0], 0.0),
        ([5.0, 5.0], 1.0),
        ([6.0, 5.0], 1.0),
        ([5.0, 6.0], 1.0),
    ]
    assert k_nearest_neighbors(data, [0.5, 0.5], 3) == 0.0
    assert k_nearest_neighbors(data, [5.5, 5.5], 3) == 1.0
    assert k_nearest_neighbors([], [1.0, 2.0], 3) is None
    # dim mismatch: distances are INF, falls back to majority of first k
    assert k_nearest_neighbors(data, [1.5], 2) == 0.0
    print("k_nearest_neighbors: OK")
