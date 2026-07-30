"""
Recursive Self-Improvement

Models of how an AI system could recursively improve its own capabilities,
leading to potentially rapid intelligence growth (seed AI / recursive
self-improvement).

https://en.wikipedia.org/wiki/Recursive_self-improvement
"""

from __future__ import annotations


class RecursiveSelfImprovementModel:
    def __init__(
        self,
        initial_intelligence: float = 1.0,
        improvement_rate: float = 0.1,
        compute_doubling_time: int = 2,
    ):
        self.intelligence = initial_intelligence
        self.improvement_rate = improvement_rate
        self.compute_doubling_time = compute_doubling_time
        self.history = [initial_intelligence]

    def step(self, iterations: int = 1) -> list[float]:
        for _ in range(iterations):
            boost = self.improvement_rate * self.intelligence
            self.intelligence += boost
            self.history.append(self.intelligence)
        return self.history

    def soft_takeoff(self, iterations: int = 20) -> list[float]:
        for i in range(iterations):
            diminishing = self.improvement_rate / (1 + 0.1 * i)
            boost = diminishing * self.intelligence
            self.intelligence += boost
            self.history.append(self.intelligence)
        return self.history

    def hard_takeoff(self, iterations: int = 20) -> list[float]:
        for _ in range(iterations):
            boost = self.improvement_rate * self.intelligence * (1 + self.intelligence / 100)
            self.intelligence += boost
            self.history.append(self.intelligence)
        return self.history


def intelligence_as_function_of_compute(
    compute: list[float], base: float = 1.0, scaling: float = 0.5
) -> list[float]:
    return [base + scaling * (c ** 0.5) for c in compute]


if __name__ == "__main__":
    model = RecursiveSelfImprovementModel(
        initial_intelligence=1.0, improvement_rate=0.2
    )
    history_linear = model.step(10)
    print("Recursive self-improvement (linear):")
    for i, val in enumerate(history_linear):
        print(f"  Iteration {i}: intelligence = {val:.3f}")

    model2 = RecursiveSelfImprovementModel(initial_intelligence=1.0, improvement_rate=0.15)
    history_soft = model2.soft_takeoff(10)
    print("\nSoft takeoff trajectory:")
    for i, val in enumerate(history_soft):
        print(f"  Iteration {i}: intelligence = {val:.3f}")

    model3 = RecursiveSelfImprovementModel(initial_intelligence=1.0, improvement_rate=0.1)
    history_hard = model3.hard_takeoff(15)
    print("\nHard takeoff trajectory (early inflection):")
    for i, val in enumerate(history_hard):
        print(f"  Iteration {i}: intelligence = {val:.3f}")

    compute = [1, 10, 100, 1000, 10000]
    iq = intelligence_as_function_of_compute(compute)
    print(f"\nCompute vs. intelligence: {list(zip(compute, [round(v, 2) for v in iq]))}")
    print("Recursive self-improvement demo complete.")
