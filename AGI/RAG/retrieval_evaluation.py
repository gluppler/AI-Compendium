"""
Retrieval Evaluation

Metrics for evaluating retrieval quality: precision, recall, 
Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG),
and Mean Average Precision (MAP).

https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
"""

from __future__ import annotations

import math


def precision_at_k(relevant: list[int], retrieved: list[int], k: int) -> float:
    retrieved_k = retrieved[:k]
    relevant_retrieved = len(set(relevant) & set(retrieved_k))
    return relevant_retrieved / k if k > 0 else 0.0


def recall_at_k(relevant: list[int], retrieved: list[int], k: int) -> float:
    retrieved_k = retrieved[:k]
    relevant_retrieved = len(set(relevant) & set(retrieved_k))
    return relevant_retrieved / len(relevant) if relevant else 0.0


def mean_reciprocal_rank(relevant_sets: list[list[int]], retrieved_list: list[list[int]]) -> float:
    reciprocal_ranks = []
    for relevant, retrieved in zip(relevant_sets, retrieved_list):
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def average_precision(relevant: list[int], retrieved: list[int]) -> float:
    ap = 0.0
    correct = 0
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            correct += 1
            ap += correct / rank
    return ap / len(relevant) if relevant else 0.0


def mean_average_precision(relevant_sets: list[list[int]], retrieved_list: list[list[int]]) -> float:
    aps = [average_prepolation(r, retrieved) for r, retrieved in zip(relevant_sets, retrieved_list)]
    return sum(aps) / len(aps) if aps else 0.0


def average_prepolation(relevant: list[int], retrieved: list[int]) -> float:
    return average_precision(relevant, retrieved)


def ndcg_at_k(relevant: list[int], retrieved: list[int], k: int, relevances: dict[int, float] | None = None) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], 1):
        rel = relevances.get(doc_id, 1.0 if doc_id in relevant else 0.0) if relevances else (1.0 if doc_id in relevant else 0.0)
        dcg += (2**rel - 1) / math.log2(rank + 1)

    ideal = sorted(
        [relevances.get(d, 0.0) if relevances else (1.0 if d in relevant else 0.0) for d in set(relevant) | set(retrieved)],
        reverse=True,
    )[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


if __name__ == "__main__":
    relevant = [1, 3, 5]
    retrieved = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(f"Precision@3: {precision_at_k(relevant, retrieved, 3):.3f}")
    print(f"Recall@3: {recall_at_k(relevant, retrieved, 3):.3f}")
    print(f"Average Precision: {average_precision(relevant, retrieved):.3f}")

    queries = [
        ([1, 2, 3], [1, 4, 5, 2, 3]),
        ([4, 5], [1, 2, 3, 4, 5]),
        ([6], [1, 2, 3, 4, 5]),
    ]
    relevant_sets = [q[0] for q in queries]
    retrieved_list = [q[1] for q in queries]
    print(f"MRR: {mean_reciprocal_rank(relevant_sets, retrieved_list):.3f}")
    print(f"MAP: {mean_average_precision(relevant_sets, retrieved_list):.3f}")

    graded = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0, 5: 2.0}
    print(f"NDCG@5: {ndcg_at_k([1, 3, 5], retrieved[:5], 5, graded):.3f}")
    print("Retrieval evaluation demo complete.")
