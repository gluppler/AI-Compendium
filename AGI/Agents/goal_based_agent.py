"""
Goal-Based Agent

A goal-based agent acts to achieve a specified goal. It combines
a model of the world with information about desired outcomes,
using search and planning to find action sequences that reach the goal.

https://en.wikipedia.org/wiki/Intelligent_agent#Goal-based_agents
"""

from __future__ import annotations

from collections import deque


class GoalBasedAgent:
    def __init__(self, initial_state: str, goal_state: str):
        self.state = initial_state
        self.goal = goal_state
        self.transitions: dict[str, dict[str, str]] = {}

    def add_transition(self, state: str, action: str, next_state: str) -> None:
        if state not in self.transitions:
            self.transitions[state] = {}
        self.transitions[state][action] = next_state

    def plan(self) -> list[str] | None:
        visited = {self.state}
        queue = deque([(self.state, [])])
        while queue:
            current, path = queue.popleft()
            if current == self.goal:
                return path
            for action, next_state in self.transitions.get(current, {}).items():
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, [*path, action]))
        return None

    def act(self, plan: list[str] | None) -> str | None:
        if plan:
            return plan[0]
        return None


if __name__ == "__main__":
    agent = GoalBasedAgent("A", "D")
    agent.add_transition("A", "go_B", "B")
    agent.add_transition("A", "go_C", "C")
    agent.add_transition("B", "go_D", "D")
    agent.add_transition("C", "go_D", "D")
    agent.add_transition("C", "go_A", "A")

    plan = agent.plan()
    print(f"Goal: {agent.goal}, Plan: {plan}")
    assert plan in (["go_B", "go_D"], ["go_C", "go_D"]), f"Unexpected plan: {plan}"
    print("Goal-based agent demo complete.")
