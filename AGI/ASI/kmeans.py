def distance(x, y):
    return sum((a - b) ** 2 for a, b in zip(x, y))

def nearest_centroids(xs, centroids):
    result = []
    for xi in xs:
        best = 0
        best_dist = float('inf')
        for i, c in enumerate(centroids):
            d = distance(xi, c)
            if d < best_dist:
                best_dist = d
                best = i
        result.append(best)
    return result

def recompute_centroids(xs, clustering, k):
    ndims = len(xs[0])
    result = []
    for cluster_ix in range(k):
        centroid = [0.0] * ndims
        n_cluster = 0.0
        for xi, zi in zip(xs, clustering):
            if zi == cluster_ix:
                n_cluster += 1.0
                for j, v in enumerate(xi):
                    centroid[j] += v
        if n_cluster > 0:
            centroid = [c / n_cluster for c in centroid]
        result.append(centroid)
    return result

def kmeans(xs, k):
    assert len(xs) >= k

    n_per_cluster = len(xs) // k
    centroids = [xs[j * n_per_cluster][:] for j in range(k)]

    clustering = nearest_centroids(xs, centroids)

    while True:
        centroids = recompute_centroids(xs, clustering, k)
        new_clustering = nearest_centroids(xs, centroids)

        if all(a == b for a, b in zip(new_clustering, clustering)):
            return clustering
        clustering = new_clustering


if __name__ == "__main__":
    xs = [[-1.1], [-1.2], [-1.3], [-1.4], [1.1], [1.2], [1.3], [1.4]]
    clustering = kmeans(xs, 2)
    assert clustering == [0, 0, 0, 0, 1, 1, 1, 1], f"got {clustering}"

    xs = [[-1.1, 0.2], [-1.2, 0.3], [-1.3, 0.1], [-1.4, 0.4],
          [1.1, -1.1], [1.2, -1.0], [1.3, -1.2], [1.4, -1.3]]
    clustering = kmeans(xs, 2)
    assert clustering == [0, 0, 0, 0, 1, 1, 1, 1], f"got {clustering}"
    print("kmeans: OK")
