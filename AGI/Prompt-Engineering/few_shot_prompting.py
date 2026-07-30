"""
Few-Shot Prompting

A prompting technique that provides a small number of input-output
examples in the prompt to guide LLM behavior without fine-tuning.

https://arxiv.org/abs/2005.14165
"""

from __future__ import annotations


class FewShotPrompt:
    def __init__(self, template: str = ""):
        self.template = template
        self.examples: list[tuple[str, str]] = []
        self.separator: str = "\n\n"

    def add_example(self, inp: str, out: str) -> None:
        self.examples.append((inp, out))

    def build(self, query: str) -> str:
        parts = [self.template] if self.template else []
        for inp, out in self.examples:
            parts.append(f"Input: {inp}\nOutput: {out}")
        parts.append(f"Input: {query}\nOutput:")
        return self.separator.join(parts)


class SentimentClassifier(FewShotPrompt):
    def __init__(self):
        super().__init__("Classify the sentiment as Positive, Negative, or Neutral.")

    def add_examples(self) -> None:
        self.add_example("I love this product!", "Positive")
        self.add_example("This is the worst movie ever.", "Negative")
        self.add_example("It's okay, nothing special.", "Neutral")
        self.add_example("Absolutely fantastic experience!", "Positive")


def dynamic_k_selection(
    query: str, example_pool: list[tuple[str, str, str]], k: int = 3
) -> list[tuple[str, str]]:
    query_words = set(query.lower().split())
    scored = []
    for inp, out, _ in example_pool:
        overlap = len(query_words & set(inp.lower().split()))
        scored.append((overlap, inp, out))
    scored.sort(key=lambda x: -x[0])
    return [(inp, out) for _, inp, out in scored[:k]]


if __name__ == "__main__":
    classifier = SentimentClassifier()
    classifier.add_examples()
    prompt = classifier.build("The battery life on this phone is terrible.")
    print(prompt)

    example_pool = [
        ("I love this product!", "Positive", "product review"),
        ("This is broken and useless.", "Negative", "product review"),
        ("It works as expected.", "Neutral", "product review"),
        ("The quality exceeded my expectations.", "Positive", "product review"),
    ]
    query = "This device stopped working after a week."
    selected = dynamic_k_selection(query, example_pool, k=2)
    print("\nDynamic K-shot selection:")
    for inp, out in selected:
        print(f"  Input: {inp} -> Output: {out}")
    print("Few-shot prompting demo complete.")
