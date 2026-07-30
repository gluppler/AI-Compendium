"""
Model-Based Agent

A model-based agent maintains an internal state (model) of the world
that is updated based on percept history. This allows it to handle
partially observable environments by tracking unobserved aspects.

https://en.wikipedia.org/wiki/Intelligent_agent#Model-based_reflex_agents
"""

from __future__ import annotations


class ModelBasedAgent:
    def __init__(self, initial_state: dict[str, object]):
        self.state = dict(initial_state)
        self.model: dict = {}
        self.rules: dict = {}

    def update_state(self, percept: object, action: str | None) -> None:
        self.state["last_percept"] = percept
        if action:
            self.state["last_action"] = action

    def act(self, percept: object) -> str | None:
        state_key = (tuple(sorted(self.state.items())), percept)
        action = self.rules.get(state_key)
        self.update_state(percept, action)
        return action


class CabbieAgent(ModelBasedAgent):
    def __init__(self):
        super().__init__({"location": "unknown", "has_passenger": False})
        self.rules = {
            (("location", "at_home"), True): "Pick up",
            (("location", "en_route"), True): "Drive to destination",
            (("location", "at_dest"), True): "Drop off",
            (("location", "at_dest"), False): "Drive to home",
        }

    def act(self, percept: dict) -> str:
        self.state.update(percept)
        return super().act(percept)


if __name__ == "__main__":
    agent = CabbieAgent()
    world = [
        {"location": "at_home", "has_passenger": True},
        {"location": "en_route", "has_passenger": True},
        {"location": "at_dest", "has_passenger": True},
        {"location": "at_dest", "has_passenger": False},
    ]
    for percept in world:
        action = agent.act(percept)
        print(f"Percept: {percept} -> Action: {action}")
    print("Model-based agent demo complete.")
