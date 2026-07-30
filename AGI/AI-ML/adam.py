class Adam:
    def __init__(self, learning_rate=None, betas=None, epsilon=None,
                 weight_decay=None, params_len=1):
        self.learning_rate = 1e-3 if learning_rate is None else learning_rate
        self.betas = (0.9, 0.999) if betas is None else betas
        self.epsilon = 1e-8 if epsilon is None else epsilon
        self.weight_decay = 0.0 if weight_decay is None else weight_decay
        self.m = [0.0] * params_len
        self.v = [0.0] * params_len
        self.t = 0

    def step(self, gradients, params):
        assert len(gradients) == len(params), \
            "gradients and params must have the same length"
        updated = [0.0] * len(params)
        self.t += 1
        b1, b2 = self.betas

        for i in range(len(gradients)):
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gradients[i]
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gradients[i] ** 2

            m_hat = self.m[i] / (1.0 - b1 ** self.t)
            v_hat = self.v[i] / (1.0 - b2 ** self.t)

            updated[i] = (params[i]
                          - self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon)
                          - self.learning_rate * self.weight_decay * params[i])
        return updated


if __name__ == "__main__":
    opt = Adam(None, None, None, None, 8)
    grads = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]
    params = [0.0] * 8
    updated = opt.step(grads, params)
    expected = [0.0009999999900000003, -0.000999999995, 0.0009999999966666666,
                -0.0009999999975, 0.000999999998, -0.0009999999983333334,
                0.0009999999985714286, -0.00099999999875]
    for u, e in zip(updated, expected):
        assert abs(u - e) < 1e-12
    print("adam: OK")
