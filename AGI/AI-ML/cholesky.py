import math

def cholesky(mat, n):
    if not mat or n == 0:
        return []
    res = [0.0] * len(mat)
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += res[i * n + k] * res[j * n + k]
            if i == j:
                diag = mat[i * n + i] - s
                res[i * n + j] = math.sqrt(diag) if diag >= 0 else 0.0
            else:
                off = mat[i * n + j] - s
                if res[j * n + j] != 0:
                    res[i * n + j] = off / res[j * n + j]
                else:
                    res[i * n + j] = 0.0
    return res


if __name__ == "__main__":
    mat = [25.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0]
    res = cholesky(mat, 3)
    expected = [5.0, 0.0, 0.0, 3.0, 3.0, 0.0, -1.0, 1.0, 3.0]
    for a, b in zip(res, expected):
        assert abs(a - b) < 1e-6
    assert cholesky([], 0) == []
    print("cholesky: OK")
