def gradient_descent(derivative_fn, x, learning_rate, num_iterations):
    for _ in range(num_iterations):
        grad = derivative_fn(x)
        for i in range(len(x)):
            x[i] -= learning_rate * grad[i]
    return x


if __name__ == "__main__":
    def deriv_square(params):
        return [2.0 * p for p in params]

    x = [5.0, 6.0]
    result = gradient_descent(deriv_square, x, 0.03, 1000)
    for v in result:
        assert abs(v) < 1e-6
    print("gradient_descent: OK")
