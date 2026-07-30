"""
Vector Search

Efficient similarity search in embedding spaces. Implements
brute-force cosine similarity search over a vector database.

https://en.wikipedia.org/wiki/Vector_database
"""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) + 1e-10
    norm_b = math.sqrt(sum(y * y for y in b)) + 1e-10
    return dot / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self):
        self.vectors: list[tuple[str, list[float]]] = []

    def add(self, doc_id: str, vector: list[float]) -> None:
        self.vectors.append((doc_id, vector))

    def add_batch(self, items: list[tuple[str, list[float]]]) -> None:
        self.vectors.extend(items)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        results = []
        for doc_id, vector in self.vectors:
            sim = cosine_similarity(query_vector, vector)
            results.append((doc_id, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def search_with_threshold(self, query_vector: list[float], threshold: float) -> list[tuple[str, float]]:
        results = []
        for doc_id, vector in self.vectors:
            sim = cosine_similarity(query_vector, vector)
            if sim >= threshold:
                results.append((doc_id, sim))
        results.sort(key=lambda x: -x[1])
        return results

    def size(self) -> int:
        return len(self.vectors)


def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


if __name__ == "__main__":
    db = VectorDatabase()
    db.add_batch([
        ("doc_1", [0.9, 0.1, 0.2, 0.3]),
        ("doc_2", [0.2, 0.8, 0.1, 0.4]),
        ("doc_3", [0.1, 0.2, 0.9, 0.1]),
        ("doc_4", [0.3, 0.4, 0.2, 0.8]),
        ("doc_5", [0.8, 0.2, 0.3, 0.2]),
    ])

    query = [0.85, 0.15, 0.25, 0.20]
    results = db.search(query, top_k=3)
    print(f"Query vector: {query}")
    print(f"Top-3 results (cosine similarity):")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")

    threshold_results = db.search_with_threshold(query, threshold=0.8)
    print(f"\nResults above 0.8 threshold: {[(d, round(s, 3)) for d, s in threshold_results]}")

    print(f"\nDatabase size: {db.size()} vectors")
    print("Vector search demo complete.")
