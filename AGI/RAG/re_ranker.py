"""
Re-Ranker

Improves retrieval quality by re-ranking initial search results
using a more expensive but more accurate model (e.g., cross-encoder).

https://www.sbert.net/examples/applications/cross-encoder/README.html
"""

from __future__ import annotations

import math


class CrossEncoder:
    def __init__(self):
        self.query_doc_pairs: list[tuple[str, str]] = []

    def predict(self, query: str, documents: list[str]) -> list[float]:
        scores = []
        for doc in documents:
            overlap = len(set(query.lower().split()) & set(doc.lower().split()))
            doc_len = len(doc.split())
            length_score = 1.0 / (1.0 + abs(len(query.split()) - doc_len) * 0.1)
            score = overlap * 0.5 + length_score * 0.5
            scores.append(min(score, 1.0))
        return scores


class ReRanker:
    def __init__(self, top_k: int = 3):
        self.model = CrossEncoder()
        self.top_k = top_k

    def rerank(self, query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        docs = [doc for doc, _ in candidates]
        scores = self.model.predict(query, docs)
        scored = [(docs[i], scores[i]) for i in range(len(docs))]
        scored.sort(key=lambda x: -x[1])
        return scored[:self.top_k]


class DiversityReRanker(ReRanker):
    def __init__(self, top_k: int = 3, diversity_lambda: float = 0.3):
        super().__init__(top_k)
        self.diversity_lambda = diversity_lambda

    def _jaccard(self, a: str, b: str) -> float:
        set_a, set_b = set(a.lower().split()), set(b.lower().split())
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union > 0 else 0.0

    def rerank(self, query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        docs = [doc for doc, _ in candidates]
        relevance = self.model.predict(query, docs)
        selected: list[tuple[str, float]] = []
        remaining = list(range(len(docs)))
        while len(selected) < self.top_k and remaining:
            best_score = float("-inf")
            best_idx = -1
            for idx in remaining:
                div_penalty = 0.0
                for sel_doc, _ in selected:
                    div_penalty = max(div_penalty, self._jaccard(docs[idx], sel_doc))
                mmr = relevance[idx] - self.diversity_lambda * div_penalty
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            if best_idx >= 0:
                selected.append((docs[best_idx], relevance[best_idx]))
                remaining.remove(best_idx)
        return selected


if __name__ == "__main__":
    query = "machine learning applications"
    candidates = [
        ("Deep learning has many applications in computer vision and NLP", 0.85),
        ("Reinforcement learning trains agents through environmental interaction", 0.45),
        ("Machine learning models are used in healthcare and finance", 0.80),
        ("The history of artificial intelligence dates back to the 1950s", 0.30),
        ("Neural networks are a fundamental building block of ML", 0.75),
    ]

    reranker = ReRanker(top_k=3)
    reranked = reranker.rerank(query, candidates)
    print("Re-ranked results (cross-encoder):")
    for doc, score in reranked:
        print(f"  {score:.3f} -> {doc}")

    diverse = DiversityReRanker(top_k=3, diversity_lambda=0.3)
    diverse_results = diverse.rerank(query, candidates)
    print("\nDiversity re-ranked (MMR):")
    for doc, score in diverse_results:
        print(f"  {score:.3f} -> {doc}")
    print("Re-ranker demo complete.")
