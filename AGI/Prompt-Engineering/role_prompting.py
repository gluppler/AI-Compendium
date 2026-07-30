"""
Role Prompting / Persona-Based Prompting

Assigning specific roles, personas, or expertise profiles to
LLMs to shape their responses and behavior.

https://arxiv.org/abs/2305.14627
"""

from __future__ import annotations


class RolePrompt:
    def __init__(self, role: str, expertise: str, tone: str = "professional"):
        self.role = role
        self.expertise = expertise
        self.tone = tone
        self.constraints: list[str] = []

    def add_constraint(self, constraint: str) -> None:
        self.constraints.append(constraint)

    def build_system_prompt(self) -> str:
        prompt = (
            f"You are a {self.role} with expertise in {self.expertise}. "
            f"Maintain a {self.tone} tone throughout."
        )
        if self.constraints:
            prompt += "\n\nConstraints:\n" + "\n".join(
                f"- {c}" for c in self.constraints
            )
        return prompt

    def build_prompt(self, user_query: str) -> str:
        return f"{self.build_system_prompt()}\n\nUser: {user_query}\nAssistant:"


class MultiRoleConversation:
    def __init__(self):
        self.participants: list[tuple[RolePrompt, str]] = []

    def add_participant(self, role: RolePrompt, name: str) -> None:
        self.participants.append((role, name))

    def simulate_exchange(self, topic: str) -> list[str]:
        responses = []
        for role, name in self.participants:
            system = role.build_system_prompt()
            response = f"[{name}] As a {role.role}, I would approach '{topic}' by..."
            responses.append(response)
        return responses


def persona_prompt_template(
    persona: str, context: str, question: str
) -> str:
    return (
        f"Persona: {persona}\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer in character:"
    )


if __name__ == "__main__":
    tutor = RolePrompt("math tutor", "algebra and calculus", "patient")
    tutor.add_constraint("Never give the answer directly, guide step by step")
    tutor.add_constraint("Use simple language")
    print("=== Role prompt ===")
    print(tutor.build_prompt("How do I solve a quadratic equation?"))

    critic = RolePrompt("code reviewer", "Python, security", "constructive")
    print("\n=== Code reviewer system prompt ===")
    print(critic.build_system_prompt())

    panel = MultiRoleConversation()
    panel.add_participant(
        RolePrompt("CEO", "business strategy", "decisive"), "Alice"
    )
    panel.add_participant(
        RolePrompt("CTO", "software engineering", "analytical"), "Bob"
    )
    print("\n=== Multi-role panel discussion ===")
    for response in panel.simulate_exchange("AI adoption strategy"):
        print(f"  {response}")

    personal_prompt = persona_prompt_template(
        "A wise old wizard from a fantasy realm",
        "The user asks about the nature of intelligence.",
        "What is consciousness?"
    )
    print(f"\n=== Persona template ===")
    print(personal_prompt)
    print("Role prompting demo complete.")
