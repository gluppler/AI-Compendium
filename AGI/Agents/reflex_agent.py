"""
Simple Reflex Agent

A simple reflex agent selects actions based solely on the current percept,
ignoring the rest of the percept history. It uses condition-action rules
(mappings from percepts to actions).

https://en.wikipedia.org/wiki/Intelligent_agent#Simple_reflex_agents
"""

from __future__ import annotations


class ReflexAgent:
    def __init__(self, rules: dict[str, str]):
        self.rules = rules

    def act(self, percept: str) -> str | None:
        return self.rules.get(percept)

    def add_rule(self, percept: str, action: str) -> None:
        self.rules[percept] = action


class VacuumWorldAgent(ReflexAgent):
    def __init__(self):
        super().__init__(
            rules={
                ("A", "Dirty"): "Suck",
                ("A", "Clean"): "Right",
                ("B", "Dirty"): "Suck",
                ("B", "Clean"): "Left",
            }
        )


if __name__ == "__main__":
    agent = VacuumWorldAgent()
    test_cases = [
        (("A", "Dirty"), "Suck"),
        (("A", "Clean"), "Right"),
        (("B", "Dirty"), "Suck"),
        (("B", "Clean"), "Left"),
    ]
    for percept, expected in test_cases:
        result = agent.act(percept)
        print(f"Percept: {percept} -> Action: {result} (expected: {expected})")
        assert result == expected, f"Failed on {percept}"
    print("All reflex agent tests passed.")
