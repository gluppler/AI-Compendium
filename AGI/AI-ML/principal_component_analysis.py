import math

def compute_means(data):
    if not data:
        return []
    num_features = len(data[0])
    means = [0.0] * num_features
    for sample in data:
        for i, v in enumerate(sample):
            means[i] += v
    n = len(data)
    return [m / n for m in means]

def center_data(data, means):
    return [[x - m for x, m in zip(sample, means)] for sample in data]

def compute_covariance_matrix(centered_data):
    if not centered_data:
        return []
    n = len(centered_data)
    num_features = len(centered_data[0])
    cov = [0.0] * (num_features * num_features)

    for i in range(num_features):
        for j in range(i, num_features):
            c = sum(sample[i] * sample[j] for sample in centered_data) / n
            cov[i * num_features + j] = c
            cov[j * num_features + i] = c
    return cov

def power_iteration(matrix, n, max_iter=1000, tolerance=1e-10):
    b_k = [1.0] * n
    b_k_prev = [0.0] * n

    for _ in range(max_iter):
        b_k_prev = b_k[:]
        b_k_new = [0.0] * n
        for i in range(n):
            for j in range(n):
                b_k_new[i] += matrix[i * n + j] * b_k[j]

        norm = math.sqrt(sum(x * x for x in b_k_new))
        if norm > 1e-10:
            b_k_new = [x / norm for x in b_k_new]
        b_k = b_k_new

        diff = max(abs(a - b) for a, b in zip(b_k, b_k_prev))
        if diff < tolerance:
            break

    eigenvalue = 0.0
    for i in range(n):
        row_sum = sum(matrix[i * n + j] * b_k[j] for j in range(n))
        eigenvalue += row_sum * b_k[i]
    eigenvalue /= sum(x * x for x in b_k)

    return eigenvalue, b_k

def deflate_matrix(matrix, eigenvector, eigenvalue, n):
    deflated = matrix[:]
    for i in range(n):
        for j in range(n):
            deflated[i * n + j] -= eigenvalue * eigenvector[i] * eigenvector[j]
    return deflated

def principal_component_analysis(data, num_components):
    if not data:
        return None
    num_features = len(data[0])
    if num_features == 0 or num_components > num_features or num_components == 0:
        return None

    means = compute_means(data)
    centered = center_data(data, means)
    cov_matrix = compute_covariance_matrix(centered)

    eigenvectors = []
    deflated = cov_matrix

    for _ in range(num_components):
        eigenvalue, eigenvector = power_iteration(deflated, num_features, 1000, 1e-10)
        eigenvectors.append(eigenvector)
        deflated = deflate_matrix(deflated, eigenvector, eigenvalue, num_features)

    transformed = []
    for sample in centered:
        transformed.append([
            sum(ev[j] * sample[j] for j in range(num_features))
            for ev in eigenvectors
        ])
    return transformed


if __name__ == "__main__":
    data = [
        [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
    ]
    result = principal_component_analysis(data, 1)
    assert result is not None
    assert len(result) == 5
    assert len(result[0]) == 1
    mean_val = sum(v[0] for v in result) / len(result)
    assert abs(mean_val) < 1e-5

    assert principal_component_analysis([], 2) is None
    print("pca: OK")
