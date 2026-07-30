"""
Self-Consistency

A decoding strategy that samples multiple reasoning chains and selects
the most consistent answer via majority voting (Wang et al., 2022).

https://arxiv.org/abs/2203.11171
"""

from __future__ import annotations

import random
from collections import Counter


class SelfConsistency:
    def __init__(self, num_chains: int = 5):
        self.num_chains = num_chains

    def generate_answer(self, question: str, chain_idx: int) -> tuple[str, str]:
        answers_pool = {
            "add": [42, 43, 41, 42, 42],
            "multiply": [120, 120, 119, 120, 121],
            "subtract": [15, 14, 15, 15, 16],
        }
        random.seed(chain_idx * 42)
        if "add" in question.lower() or "sum" in question.lower():
            answer = random.choice(answers_pool["add"])
            reasoning = f"Adding the numbers gives {answer}."
        elif "multiply" in question.lower() or "product" in question.lower():
            answer = random.choice(answers_pool["multiply"])
            reasoning = f"Multiplying the numbers gives {answer}."
        elif "subtract" in question.lower() or "difference" in question.lower():
            answer = random.choice(answers_pool["subtract"])
            reasoning = f"Subtracting the numbers gives {answer}."
        else:
            answer = 0
            reasoning = "Unknown operation."
        return reasoning, str(answer)

    def solve(self, question: str) -> tuple[str, str, list[str]]:
        answers: list[str] = []
        reasonings: list[str] = []
        for i in range(self.num_chains):
            reasoning, answer = self.generate_answer(question, i)
            answers.append(answer)
            reasonings.append(reasoning)

        counter = Counter(answers)
        most_common = counter.most_common(1)[0][0]
        return most_common, answers, reasonings

    def confidence(self, answers: list[str]) -> float:
        if not answers:
            return 0.0
        counter = Counter(answers)
        top_count = counter.most_common(1)[0][1]
        return top_count / len(answers)


def weighted_majority_vote(
    answers: list[str], weights: list[float]
) -> str:
    weighted: dict[str, float] = {}
    for ans, w in zip(answers, weights):
        weighted[ans] = weighted.get(ans, 0.0) + w
    return max(weighted, key=weighted.get)


if __name__ == "__main__":
    sc = SelfConsistency(num_chains=7)
    question = "What is the sum of 20 and 22?"
    final_answer, all_answers, reasonings = sc.solve(question)
    print(f"Question: {question}")
    print(f"Individual answers: {all_answers}")
    print(f"Final (majority vote): {final_answer}")
    print(f"Confidence: {sc.confidence(all_answers):.2f}")

    weighted = weighted_majority_vote(
        all_answers, [1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7]
    )
    print(f"Weighted majority: {weighted}")
    print("Self-consistency demo complete.")
