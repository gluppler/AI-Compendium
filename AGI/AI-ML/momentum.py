def momentum(derivative, x, learning_rate, beta, num_iterations):
    velocity = [0.0] * len(x)
    for _ in range(num_iterations):
        grad = derivative(x)
        for i in range(len(x)):
            velocity[i] = beta * velocity[i] + grad[i]
            x[i] -= learning_rate * velocity[i]
    return x


if __name__ == "__main__":
    def deriv_square(params):
        return [2.0 * p for p in params]

    x = [5.0, 6.0]
    result = momentum(deriv_square, x, 0.01, 0.9, 1000)
    for v in result:
        assert abs(v) < 1e-6
    print("momentum: OK")
