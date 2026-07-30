"""
Utility-Based Agent

A utility-based agent chooses actions that maximize its expected
utility (happiness/performance measure). Unlike goal-based agents
(binary success/failure), utility agents handle trade-offs and
uncertain outcomes by preferring higher-utility states.

https://en.wikipedia.org/wiki/Intelligent_agent#Utility-based_agents
"""

from __future__ import annotations


class UtilityBasedAgent:
    def __init__(self, utility_function: dict[str, float]):
        self.utility = dict(utility_function)
        self.transitions: dict[str, dict[str, list[tuple[str, float]]]] = {}

    def add_action(
        self, state: str, action: str, outcomes: list[tuple[str, float]]
    ) -> None:
        if state not in self.transitions:
            self.transitions[state] = {}
        self.transitions[state][action] = outcomes

    def expected_utility(self, state: str, action: str) -> float:
        total = 0.0
        for next_state, prob in self.transitions.get(state, {}).get(action, []):
            total += prob * self.utility.get(next_state, 0.0)
        return total

    def best_action(self, state: str) -> tuple[str | None, float]:
        best_action = None
        best_value = float("-inf")
        for action in self.transitions.get(state, {}):
            eu = self.expected_utility(state, action)
            if eu > best_value:
                best_value = eu
                best_action = action
        return best_action, best_value


if __name__ == "__main__":
    agent = UtilityBasedAgent(
        utility_function={"Sunny": 10, "Rainy": -5, "Home": 2}
    )
    agent.add_action("Start", "Walk", [("Sunny", 0.6), ("Rainy", 0.4)])
    agent.add_action("Start", "Take Bus", [("Sunny", 0.9), ("Rainy", 0.1)])

    for state in ["Start"]:
        action, value = agent.best_action(state)
        print(
            f"State: {state} -> Best action: {action} "
            f"(expected utility: {value:.2f})"
        )
    print("Utility-based agent demo complete.")
