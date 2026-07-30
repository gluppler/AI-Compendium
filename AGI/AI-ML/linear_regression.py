import math

def linear_regression(data_points):
    if not data_points:
        return None
    n = len(data_points)
    mean_x = sum(p[0] for p in data_points) / n
    mean_y = sum(p[1] for p in data_points) / n

    covariance = 0.0
    std_dev_sqr_x = 0.0
    std_dev_sqr_y = 0.0

    for x, y in data_points:
        covariance += (x - mean_x) * (y - mean_y)
        std_dev_sqr_x += (x - mean_x) ** 2
        std_dev_sqr_y += (y - mean_y) ** 2

    std_dev_x = math.sqrt(std_dev_sqr_x)
    std_dev_y = math.sqrt(std_dev_sqr_y)
    std_dev_prod = std_dev_x * std_dev_y

    pcc = covariance / std_dev_prod
    b = pcc * (std_dev_y / std_dev_x)
    a = mean_y - b * mean_x
    return (a, b)


if __name__ == "__main__":
    data = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    result = linear_regression(data)
    assert result is not None
    a, b = result
    assert abs(a) < 1e-10
    assert abs(b - 1.0) < 1e-10

    assert linear_regression([]) is None
    print("linear_regression: OK")
