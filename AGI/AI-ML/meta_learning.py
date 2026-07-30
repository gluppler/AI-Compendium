"""
Meta-Learning (MAML)

Model-Agnostic Meta-Learning (Finn et al., 2017) enables models to
learn new tasks from a small number of gradient steps. This implements
the inner-loop (task-specific) and outer-loop (meta) update procedure.

https://arxiv.org/abs/1703.03400
"""

from __future__ import annotations

import math
import random


def sine_function(a: float, b: float, x: float) -> float:
    return a * math.sin(x + b)


def maml_inner_loop(
    x_train: list[float],
    y_train: list[float],
    init_params: tuple[float, float],
    lr: float = 0.01,
    steps: int = 5,
) -> tuple[float, float]:
    a, b = init_params
    for _ in range(steps):
        grad_a = 0.0
        grad_b = 0.0
        for x, y in zip(x_train, y_train):
            pred = sine_function(a, b, x)
            error = pred - y
            grad_a += 2 * error * math.sin(x + b)
            grad_b += 2 * error * a * math.cos(x + b)
        n = len(x_train)
        a -= lr * (grad_a / n)
        b -= lr * (grad_b / n)
    return a, b


def maml_outer_loop(
    tasks: list[tuple[list[float], list[float], list[float], list[float]]],
    meta_lr: float = 0.001,
    inner_lr: float = 0.01,
    inner_steps: int = 5,
    epochs: int = 20,
) -> tuple[float, float]:
    meta_params = [random.uniform(0.5, 1.5), random.uniform(-0.5, 0.5)]
    for _ in range(epochs):
        grad_a, grad_b = 0.0, 0.0
        for x_train, y_train, x_val, y_val in tasks:
            adapted = maml_inner_loop(x_train, y_train, tuple(meta_params), inner_lr, inner_steps)
            val_error = 0.0
            for x, y in zip(x_val, y_val):
                pred = sine_function(*adapted, x)
                val_error += (pred - y) ** 2
            for x, y in zip(x_val, y_val):
                pred = sine_function(*adapted, x)
                error = pred - y
                grad_a += 2 * error * math.sin(x + adapted[1])
                grad_b += 2 * error * adapted[0] * math.cos(x + adapted[1])
        meta_params[0] -= meta_lr * (grad_a / len(tasks))
        meta_params[1] -= meta_lr * (grad_b / len(tasks))
    return tuple(meta_params)


if __name__ == "__main__":
    random.seed(42)
    tasks = []
    for _ in range(8):
        amp = random.uniform(0.5, 2.0)
        phase = random.uniform(-1.0, 1.0)
        xs = [random.uniform(-5, 5) for _ in range(10)]
        ys = [sine_function(amp, phase, x) for x in xs]
        xs_val = [random.uniform(-5, 5) for _ in range(5)]
        ys_val = [sine_function(amp, phase, x) for x in xs_val]
        tasks.append((xs, ys, xs_val, ys_val))

    meta_params = maml_outer_loop(tasks)
    print(f"Meta-learned initialization: a={meta_params[0]:.3f}, b={meta_params[1]:.3f}")

    new_x = [random.uniform(-5, 5) for _ in range(5)]
    new_y = [sine_function(1.2, 0.3, x) for x in new_x]
    adapted = maml_inner_loop(new_x, new_y, meta_params, lr=0.05, steps=10)
    print(f"After 10-shot adaptation: a={adapted[0]:.3f}, b={adapted[1]:.3f}")
    print("Meta-learning (MAML) demo complete.")
