"""
Context Compression

Techniques for compressing context to fit within limited windows:
extractive summarization, selective context dropping, and
information-dense reformatting.

https://arxiv.org/abs/2307.06436
"""

from __future__ import annotations


class ContextCompressor:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def extractive_summarize(self, text: str, top_n: int = 3) -> str:
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if len(sentences) <= top_n:
            return text
        scored = []
        for s in sentences:
            score = len(s.split())
            keywords = sum(1 for w in ["important", "key", "critical", "result", "conclusion", "significant"]
                           if w in s.lower())
            score += keywords * 3
            scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        selected = [s for _, s in scored[:top_n]]
        return ". ".join(selected) + "."

    def keyword_preserving_compression(self, text: str, keywords: list[str]) -> str:
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        kept = []
        for s in sentences:
            if any(k.lower() in s.lower() for k in keywords):
                kept.append(s)
        return ". ".join(kept) + "."

    def drop_redundant(self, messages: list[str], similarity_threshold: float = 0.7) -> list[str]:
        deduped: list[str] = []
        for msg in messages:
            is_duplicate = False
            for existing in deduped:
                if self._jaccard_similarity(msg, existing) > similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(msg)
        return deduped

    def _jaccard_similarity(self, a: str, b: str) -> float:
        set_a, set_b = set(a.lower().split()), set(b.lower().split())
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    compressor = ContextCompressor(max_tokens=100)

    long_text = (
        "The experiment produced several important results. "
        "The key finding was that model performance improved with scale. "
        "We also observed diminishing returns beyond a certain threshold. "
        "The critical conclusion is that architecture matters more than size. "
        "Temperature settings affected output diversity significantly. "
        "Some secondary observations about training stability were noted."
    )

    summary = compressor.extractive_summarize(long_text, top_n=2)
    print(f"Extractive summary: {summary}")

    kw_compressed = compressor.keyword_preserving_compression(
        long_text, ["important", "key", "critical", "conclusion"]
    )
    print(f"\nKeyword-preserved: {kw_compressed}")

    messages = [
        "The model performed well on the test set.",
        "The model performed well on the test set.",
        "Performance on the validation set was also good.",
        "Training loss converged after 100 epochs.",
    ]
    deduped = compressor.drop_redundant(messages)
    print(f"\nDeduplicated messages ({len(deduped)} from {len(messages)}):")
    for m in deduped:
        print(f"  - {m}")
    print("Context compression demo complete.")
