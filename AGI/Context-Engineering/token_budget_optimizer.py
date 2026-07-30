"""
Token Budget Optimizer

Strategies for allocating limited token budgets across prompt components
(system, conversation, context, tools) to maximize task performance.

https://platform.openai.com/docs/guides/text-generation/managing-tokens
"""

from __future__ import annotations


class TokenBudgetOptimizer:
    def __init__(self, total_budget: int):
        self.total = total_budget
        self.minimums: dict[str, int] = {}
        self.weights: dict[str, float] = {}

    def set_minimum(self, component: str, tokens: int) -> None:
        self.minimums[component] = tokens

    def set_weight(self, component: str, weight: float) -> None:
        self.weights[component] = weight

    def optimize(self) -> dict[str, int]:
        total_min = sum(self.minimums.values())
        remaining = self.total - total_min
        if remaining <= 0:
            total_min = sum(self.minimums.values())
            scale = self.total / total_min if total_min > 0 else 1
            return {k: int(v * scale) for k, v in self.minimums.items()}

        total_weight = sum(self.weights.values())
        allocation = dict(self.minimums)
        for component, weight in self.weights.items():
            extra = int(remaining * (weight / total_weight))
            allocation[component] = allocation.get(component, 0) + extra

        used = sum(allocation.values())
        if used != self.total:
            diff = self.total - used
            if diff > 0:
                heaviest = max(allocation, key=lambda k: allocation[k])
                allocation[heaviest] += diff
            elif diff < 0:
                lightest = min(allocation, key=lambda k: allocation[k])
                allocation[lightest] = max(0, allocation[lightest] + diff)

        return allocation


def truncate_to_budget(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def estimate_tokens(text: str) -> int:
    return len(text.split()) + int(len(text) * 0.25)


if __name__ == "__main__":
    optimizer = TokenBudgetOptimizer(total_budget=4096)
    optimizer.set_minimum("system", 300)
    optimizer.set_minimum("history", 500)
    optimizer.set_minimum("rag_context", 200)
    optimizer.set_weight("system", 2.0)
    optimizer.set_weight("history", 1.0)
    optimizer.set_weight("rag_context", 3.0)
    optimizer.set_weight("tools", 1.5)

    allocation = optimizer.optimize()
    print("Optimized token allocation:")
    for component, tokens in allocation.items():
        print(f"  {component}: {tokens} tokens")

    text = "This is a sample long text that " * 100
    truncated = truncate_to_budget(text, 10)
    print(f"\nTruncation: '{text[:50]}...' -> '{truncated}'")
    print(f"Estimated tokens in short text: {estimate_tokens(truncated)}")
    print("Token budget optimizer demo complete.")
