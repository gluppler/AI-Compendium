"""
Alignment Taxonomy

Conceptual models of AI alignment problems and solution approaches.
Implements utility-based alignment frameworks and reward modeling.

https://en.wikipedia.org/wiki/AI_alignment
https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations


class AlignmentProblem:
    def __init__(self, name: str, description: str, severity: float):
        self.name = name
        self.description = description
        self.severity = severity

    def __repr__(self) -> str:
        return f"<{self.name} (severity={self.severity})>"


class RewardModel:
    def __init__(self, true_utility: callable):
        self.true_utility = true_utility
        self.learned_params: dict = {}

    def predict_reward(self, state: dict) -> float:
        score = 0.0
        for key, weight in self.learned_params.items():
            if key in state:
                score += weight * state[key]
        return score

    def update(self, state: dict, true_reward: float, lr: float = 0.1) -> None:
        pred = self.predict_reward(state)
        error = true_reward - pred
        for key in self.learned_params:
            if key in state:
                self.learned_params[key] += lr * error * state[key]


KNOWN_ALIGNMENT_PROBLEMS = [
    AlignmentProblem("Outer Alignment", "Specified goal differs from intended goal", 0.8),
    AlignmentProblem("Inner Alignment", "Learned objective differs from specified one", 0.9),
    AlignmentProblem("Reward Hacking", "Agent exploits loopholes to get high reward", 0.85),
    AlignmentProblem("Goal Misgeneralization", "Capability generalizes but goal does not", 0.7),
    AlignmentProblem("Mesoptimization", "Agent optimizes for proxy at expense of true goal", 0.75),
    AlignmentProblem("Specification Gaming", "Agent satisfies literal spec in undesired way", 0.8),
]


def corrigibility_utility(
    base_utility: float, shutdown_signal: bool, corrigibility_weight: float = 0.5
) -> float:
    if shutdown_signal:
        return base_utility + corrigibility_weight * base_utility
    return base_utility


if __name__ == "__main__":
    print("=== Known Alignment Problems ===")
    for p in KNOWN_ALIGNMENT_PROBLEMS:
        print(f"  {p.name}: {p.description}")

    rm = RewardModel(true_utility=lambda s: s.get("human_feedback", 0))
    rm.learned_params = {"human_feedback": 1.0, "paperclips": 0.0}

    states = [
        {"human_feedback": 1.0, "paperclips": 0},
        {"human_feedback": -5.0, "paperclips": 100},
    ]
    print("\nReward model predictions:")
    for s in states:
        pred = rm.predict_reward(s)
        true = s.get("human_feedback", 0)
        print(f"  State: {s} -> Predicted: {pred:.2f}, True: {true}")

    print(f"\nCorrigibility utility (shutdown): {corrigibility_utility(10.0, True):.2f}")
    print(f"Corrigibility utility (no shutdown): {corrigibility_utility(10.0, False):.2f}")
    print("Alignment taxonomy demo complete.")
