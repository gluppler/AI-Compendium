"""
Context Window Manager

Strategies for managing LLM context windows: sliding windows,
rolling windows, and priority-based eviction for fitting
information within token budgets.

https://en.wikipedia.org/wiki/Context_window
"""

from __future__ import annotations

from collections import deque


class SlidingWindow:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.tokens: deque[str] = deque()

    def add(self, token: str) -> list[str]:
        self.tokens.append(token)
        while len(self.tokens) > self.max_tokens:
            self.tokens.popleft()
        return list(self.tokens)

    def add_batch(self, tokens: list[str]) -> list[str]:
        for t in tokens:
            self.tokens.append(t)
        while len(self.tokens) > self.max_tokens:
            self.tokens.popleft()
        return list(self.tokens)

    def content(self) -> list[str]:
        return list(self.tokens)


class PriorityContext:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.entries: list[tuple[str, float]] = []

    def add(self, text: str, priority: float) -> None:
        self.entries.append((text, priority))
        self.entries.sort(key=lambda x: -x[1])

    def get_context(self, max_tokens: int | None = None) -> str:
        limit = max_tokens or self.max_tokens
        result = []
        used = 0
        for text, _ in self.entries:
            tokens = text.split()
            if used + len(tokens) <= limit:
                result.append(text)
                used += len(tokens)
        return "\n".join(result)


class TokenBudget:
    def __init__(self, total_budget: int):
        self.total = total_budget
        self.allocations: dict[str, int] = {}

    def allocate(self, component: str, tokens: int) -> bool:
        used = sum(self.allocations.values())
        if used + tokens <= self.total:
            self.allocations[component] = tokens
            return True
        return False

    def remaining(self) -> int:
        return self.total - sum(self.allocations.values())

    def summary(self) -> dict:
        return {
            "total": self.total,
            "used": sum(self.allocations.values()),
            "remaining": self.remaining(),
            "allocations": dict(self.allocations),
        }


if __name__ == "__main__":
    sw = SlidingWindow(5)
    for token in ["the", "quick", "brown", "fox", "jumps", "over", "the", "dog"]:
        sw.add(token)
    print(f"Sliding window content: {sw.content()}")

    pc = PriorityContext(20)
    pc.add("System prompt with instructions.", priority=10.0)
    pc.add("User conversation history.", priority=5.0)
    pc.add("Retrieved document context.", priority=7.0)
    print(f"\nPriority context:\n{pc.get_context()}")

    budget = TokenBudget(4096)
    budget.allocate("system", 500)
    budget.allocate("history", 2000)
    budget.allocate("rag_context", 1000)
    print(f"\nToken budget: {budget.summary()}")
    print("Context window management demo complete.")
