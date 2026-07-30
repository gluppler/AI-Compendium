import math

class SVCError(Exception):
    pass

class Kernel:
    Linear = "linear"
    Rbf = "rbf"

class SVC:
    def __init__(self, kernel="linear", regularization=float('inf'), gamma=1.0):
        if regularization <= 0.0:
            raise SVCError("regularization must be > 0")
        if kernel == Kernel.Rbf and gamma <= 0.0:
            raise SVCError("gamma must be > 0")

        self.kernel = kernel
        self.regularization = regularization
        self.gamma = gamma
        self.observations = []
        self.classes = []
        self.optimum = []
        self.offset = 0.0

    def kernel_function(self, v1, v2):
        if self.kernel == Kernel.Linear:
            return sum(a * b for a, b in zip(v1, v2))
        elif self.kernel == Kernel.Rbf:
            diff = sum((a - b) ** 2 for a, b in zip(v1, v2))
            return math.exp(-self.gamma * diff)

    def fit(self, observations, classes):
        if not observations or not classes:
            raise SVCError("Empty data")
        if len(observations) != len(classes):
            raise SVCError("Mismatched dimensions")

        self.observations = [o[:] for o in observations]
        self.classes = classes[:]
        n = len(classes)

        self.optimum = self.solve_dual(n)
        self.offset = self.calculate_offset(n)

    def solve_dual(self, n):
        lam = [0.5] * n
        learning_rate = 0.1
        iterations = 5000
        tolerance = 1e-8

        kernel_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                kernel_matrix[i][j] = self.kernel_function(
                    self.observations[i], self.observations[j])

        for iteration in range(iterations):
            gradient = [0.0] * n
            for i in range(n):
                s = 0.0
                for j in range(n):
                    s += lam[j] * self.classes[j] * kernel_matrix[i][j]
                gradient[i] = self.classes[i] * s - 1.0

            old_lam = lam[:]
            lr = learning_rate / (1.0 + iteration / 1000.0)
            lam = [lam[i] - lr * gradient[i] for i in range(n)]

            for i in range(n):
                lam[i] = max(0.0, min(self.regularization, lam[i]))

            # Enforce sum(lambda * y) = 0
            sum_ly = sum(lam[i] * self.classes[i] for i in range(n))
            correction = sum_ly / n
            for _ in range(10):
                for i in range(n):
                    delta = correction * self.classes[i]
                    new_val = lam[i] - delta
                    lam[i] = max(0.0, min(self.regularization, new_val))
                sum_ly = sum(lam[i] * self.classes[i] for i in range(n))
                correction = sum_ly / n
                if abs(sum_ly) < 1e-10:
                    break

            diff = math.sqrt(sum((lam[i] - old_lam[i]) ** 2 for i in range(n)))
            if diff < tolerance:
                break

        return lam

    def calculate_offset(self, n):
        s = 0.0
        count = 0
        threshold = 1e-5

        for i in range(n):
            if (self.optimum[i] > threshold
                    and self.optimum[i] < self.regularization - threshold):
                kernel_sum = 0.0
                for j in range(n):
                    kernel_sum += (self.optimum[j] * self.classes[j]
                                   * self.kernel_function(self.observations[j],
                                                          self.observations[i]))
                s += self.classes[i] - kernel_sum
                count += 1

        if count == 0:
            for i in range(n):
                kernel_sum = 0.0
                for j in range(n):
                    kernel_sum += (self.optimum[j] * self.classes[j]
                                   * self.kernel_function(self.observations[j],
                                                          self.observations[i]))
                s += self.classes[i] - kernel_sum
            return s / n
        else:
            return s / count

    def predict(self, observation):
        s = 0.0
        for i in range(len(self.classes)):
            s += (self.optimum[i] * self.classes[i]
                  * self.kernel_function(self.observations[i], observation))
        return 1.0 if s + self.offset >= 0.0 else -1.0

    def n_support_vectors(self):
        return sum(1 for l in self.optimum if l > 1e-5)


if __name__ == "__main__":
    obs = [[0.0, 1.0], [0.0, 2.0], [1.0, 1.0], [1.0, 2.0]]
    cls = [1.0, 1.0, -1.0, -1.0]

    svc = SVC(Kernel.Linear, float('inf'))
    svc.fit(obs, cls)

    assert svc.predict([0.0, 1.0]) == 1.0
    assert svc.predict([1.0, 1.0]) == -1.0
    assert svc.n_support_vectors() > 0
    print("svc: OK")
