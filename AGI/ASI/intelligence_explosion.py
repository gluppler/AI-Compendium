"""
Intelligence Explosion / FOOM

Models of rapid AI capability growth scenarios, including
soft vs. hard takeoff dynamics from AI risk literature.

https://en.wikipedia.org/wiki/Intelligence_explosion
"""

from __future__ import annotations

import math


class IntelligenceExplosionModel:
    def __init__(
        self,
        seed_iq: float = 1.0,
        recursion_efficiency: float = 0.3,
        hardware_growth: float = 0.05,
    ):
        self.iq = seed_iq
        self.recursion_efficiency = recursion_efficiency
        self.hardware_growth = hardware_growth
        self.history: list[tuple[int, float, float]] = []

    def step(self, iterations: int = 30) -> list[tuple[int, float, float]]:
        hardware = 1.0
        for t in range(iterations):
            rsi_boost = self.recursion_efficiency * self.iq
            hardware *= 1 + self.hardware_growth
            self.iq += rsi_boost * hardware
            self.history.append((t, self.iq, hardware))
        return self.history

    def crossover_timestep(self, threshold: float = 10.0) -> int | None:
        for t, iq, _ in self.history:
            if iq >= threshold:
                return t
        return None


def compute_doubling_periods(initial: float, growth_rate: float, periods: int) -> list[float]:
    values = [initial]
    for _ in range(periods):
        values.append(values[-1] * (1 + growth_rate))
    return values


if __name__ == "__main__":
    model = IntelligenceExplosionModel(
        seed_iq=1.0, recursion_efficiency=0.2, hardware_growth=0.03
    )
    history = model.step(25)
    print("Intelligence explosion trajectory:")
    for t, iq, hw in history:
        print(f"  t={t:2d}: IQ={iq:.2f}, Hardware={hw:.2f}x")

    crossover = model.crossover_timestep(10.0)
    print(f"\nCrossover (IQ >= 10x human): t={crossover}")

    doublings = compute_doubling_periods(1.0, 0.5, 10)
    print(f"\nDoubling periods: {[round(d, 2) for d in doublings]}")
    print("Intelligence explosion demo complete.")
