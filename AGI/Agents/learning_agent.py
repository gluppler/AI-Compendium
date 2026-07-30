"""
Learning Agent

A learning agent improves its performance over time by learning
from experience. It has four conceptual components: the learning
element (improves knowledge), the performance element (selects actions),
the critic (provides feedback), and the problem generator (suggests
exploratory actions).

https://en.wikipedia.org/wiki/Intelligent_agent#Learning_agents
"""

from __future__ import annotations

import random


class LearningAgent:
    def __init__(self, actions: list[str], epsilon: float = 0.1, alpha: float = 0.5):
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_table: dict[tuple, list[float]] = {}

    def get_q(self, state: tuple) -> list[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)
        return self.q_table[state]

    def act(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randrange(len(self.actions))
        q_values = self.get_q(state)
        max_q = max(q_values)
        best = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best)

    def learn(
        self, state: tuple, action_idx: int, reward: float, next_state: tuple
    ) -> None:
        q = self.get_q(state)
        next_max = max(self.get_q(next_state))
        q[action_idx] += self.alpha * (
            reward + 0.9 * next_max - q[action_idx]
        )


if __name__ == "__main__":
    agent = LearningAgent(["left", "right"])
    states = [(i,) for i in range(5)]
    for episode in range(100):
        state = random.choice(states)
        for _ in range(10):
            action_idx = agent.act(state)
            action = agent.actions[action_idx]
            next_state = (
                max(0, state[0] - 1) if action == "left"
                else min(4, state[0] + 1)
            )
            reward = 1.0 if next_state == 4 else 0.0
            agent.learn(state, action_idx, reward, (next_state,))
            state = (next_state,)

    print("Q-table after training:")
    for state in states:
        print(f"  State {state[0]}: {[f'{q:.2f}' for q in agent.get_q(state)]}")
    print("Learning agent demo complete.")
