"""
Tree of Thoughts (ToT)

A prompting framework that explores multiple reasoning paths
simultaneously using a tree search over intermediate "thoughts"
(Yao et al., 2023).

https://arxiv.org/abs/2305.10601
"""

from __future__ import annotations

from collections import deque


class ThoughtNode:
    def __init__(self, thought: str, parent: ThoughtNode | None = None):
        self.thought = thought
        self.parent = parent
        self.children: list[ThoughtNode] = []
        self.value: float = 0.0

    def add_child(self, child: ThoughtNode) -> None:
        self.children.append(child)


class TreeOfThoughts:
    def __init__(self, branching_factor: int = 3, max_depth: int = 5):
        self.branching_factor = branching_factor
        self.max_depth = max_depth

    def generate_thoughts(self, state: str, step: int) -> list[str]:
        candidates = [
            f"[Step {step}] Consider option A: {state} -> split into subproblems",
            f"[Step {step}] Consider option B: {state} -> try direct approach",
            f"[Step {step}] Consider option C: {state} -> backtrack and retry",
        ]
        return candidates[:self.branching_factor]

    def evaluate_thought(self, thought: str) -> float:
        promising = ["split", "direct", "retry", "subproblems"]
        score = 0.5
        for word in promising:
            if word in thought.lower():
                score += 0.1
        return min(score, 1.0)

    def bfs_search(self, initial_state: str) -> list[str]:
        root = ThoughtNode(initial_state)
        queue = deque([root])
        best_path: list[str] = []
        best_value = float("-inf")

        while queue:
            node = queue.popleft()
            depth = self._depth(node)
            if depth >= self.max_depth:
                continue

            thoughts = self.generate_thoughts(node.thought, depth)
            for thought_text in thoughts:
                child = ThoughtNode(thought_text, node)
                child.value = self.evaluate_thought(thought_text)
                node.add_child(child)
                queue.append(child)

                if child.value > best_value:
                    best_value = child.value
                    best_path = self._path_to_root(child)
        return best_path

    def _depth(self, node: ThoughtNode) -> int:
        d = 0
        while node.parent:
            d += 1
            node = node.parent
        return d

    def _path_to_root(self, node: ThoughtNode) -> list[str]:
        path = []
        while node:
            path.append(node.thought)
            node = node.parent
        return list(reversed(path))


if __name__ == "__main__":
    tot = TreeOfThoughts(branching_factor=3, max_depth=4)
    path = tot.bfs_search("Solve: 24 game with numbers [4, 7, 8, 8]")
    print("Best path found (BFS):")
    for step in path:
        print(f"  {step}")
    print(f"Path length: {len(path)}")
    print("Tree of Thoughts demo complete.")
