import math
import random

def get_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)

def find_nearest(data_point, centroids):
    nearest = 0
    min_dist = get_distance(data_point, centroids[0])
    for i, c in enumerate(centroids):
        d = get_distance(data_point, c)
        if d < min_dist:
            min_dist = d
            nearest = i
    return nearest

def k_means(data_points, n_clusters, max_iter=100):
    if len(data_points) < n_clusters:
        return None

    centroids = [(random.random(), random.random()) for _ in range(n_clusters)]
    labels = [0] * len(data_points)

    for _ in range(max_iter):
        new_positions = [(0.0, 0.0) for _ in range(n_clusters)]
        new_counts = [0] * n_clusters

        for i, d in enumerate(data_points):
            nearest = find_nearest(d, centroids)
            labels[i] = nearest
            new_positions[nearest] = (new_positions[nearest][0] + d[0],
                                       new_positions[nearest][1] + d[1])
            new_counts[nearest] += 1

        for i in range(n_clusters):
            if new_counts[i] > 0:
                centroids[i] = (new_positions[i][0] / new_counts[i],
                                new_positions[i][1] / new_counts[i])

    return labels


if __name__ == "__main__":
    data = [(1.0, 1.0), (1.1, 1.0), (1.0, 1.1),
            (5.0, 5.0), (5.1, 5.0), (5.0, 5.1)]
    labels = k_means(data, 2, 100)
    assert labels is not None
    assert len(labels) == 6
    # Points 0-2 should be in same cluster, 3-5 in the other
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    print("k_means: OK")
