"""
Chain of Thought Prompting

A prompting technique that encourages LLMs to reason step-by-step,
improving performance on complex reasoning tasks (Wei et al., 2022).

https://arxiv.org/abs/2201.11903
"""

from __future__ import annotations

import re


def chain_of_thought_prompt(question: str, examples: list[tuple[str, str, str]] | None = None) -> str:
    prompt = ""
    if examples:
        prompt += "Solve these problems step by step:\n\n"
        for q, reasoning, answer in examples:
            prompt += f"Q: {q}\nA: {reasoning}\nTherefore, the answer is {answer}.\n\n"
    prompt += f"Q: {question}\nA: Let's work through this step by step."
    return prompt


class ChainOfThoughtSolver:
    def __init__(self):
        self.examples: list[tuple[str, str, str]] = []

    def add_example(self, question: str, reasoning: str, answer: str) -> None:
        self.examples.append((question, reasoning, answer))

    def solve(self, question: str) -> str:
        return chain_of_thought_prompt(question, self.examples)


def extract_answer_from_cot(text: str) -> str | None:
    patterns = [
        r"answer is (.+?)\.", r"answer: (.+)", r"so (.+) is the answer",
        r"therefore, (.+)", r"final answer: (.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()
    return None


if __name__ == "__main__":
    solver = ChainOfThoughtSolver()
    solver.add_example(
        "If a store has 12 apples and sells 4, how many remain?",
        "Start with 12 apples. Sell 4 means subtract: 12 - 4 = 8.",
        "8"
    )
    solver.add_example(
        "A train travels 60 miles per hour for 2 hours. How far does it go?",
        "Distance = speed x time. Speed = 60 mph, time = 2 hours. 60 x 2 = 120.",
        "120 miles"
    )

    question = "A farmer has 15 chickens. Each chicken lays 2 eggs per day. How many eggs in 3 days?"
    prompt = solver.solve(question)
    print(prompt)

    sample_reasoning = "Each chicken lays 2 eggs per day. 15 chickens * 2 eggs = 30 eggs per day. Over 3 days: 30 * 3 = 90. Therefore, the answer is 90 eggs."
    extracted = extract_answer_from_cot(sample_reasoning)
    print(f"\nExtracted answer: {extracted}")
    print("Chain-of-thought prompting demo complete.")
