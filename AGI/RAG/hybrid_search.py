"""
Hybrid Search

Combines dense (embedding-based) and sparse (keyword-based) search
to improve retrieval quality. Uses reciprocal rank fusion (RRF)
to merge results from both methods.

https://www.pinecone.io/learn/hybrid-search/
"""

from __future__ import annotations

import math


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq: dict[str, int] = {}
        self.doc_lengths: list[int] = []
        self.documents: list[str] = []
        self.avg_doc_length: float = 0.0

    def fit(self, documents: list[str]) -> None:
        self.documents = documents
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(documents), 1)
        term_doc_counts: dict[str, set] = {}
        for doc in documents:
            for term in set(doc.lower().split()):
                if term not in term_doc_counts:
                    term_doc_counts[term] = set()
                term_doc_counts[term].add(doc)
        self.doc_freq = {term: len(docs) for term, docs in term_doc_counts.items()}

    def score(self, query: str, doc_idx: int) -> float:
        doc = self.documents[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        score = 0.0
        n_docs = len(self.documents)
        for term in query.lower().split():
            if term not in self.doc_freq:
                continue
            idf = math.log((n_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5) + 1)
            tf = doc.lower().split().count(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        scores = [(self.documents[i], self.score(query, i)) for i in range(len(self.documents))]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(dense_results):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(sparse_results):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    result = sorted(fused.items(), key=lambda x: -x[1])
    return result


class HybridSearch:
    def __init__(self, dense_weight: float = 0.5):
        self.bm25 = BM25()
        self.dense_weight = dense_weight

    def search(self, query: str, dense_results: list[tuple[str, float]], top_k: int = 5) -> list[tuple[str, float]]:
        sparse_results = self.bm25.search(query, top_k * 2)
        return reciprocal_rank_fusion(dense_results, sparse_results)[:top_k]


if __name__ == "__main__":
    bm25 = BM25()
    docs = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with many layers",
        "retrieval augmented generation enhances llms with external knowledge",
        "vector databases enable similarity search on embeddings",
        "natural language processing is used for text understanding",
    ]
    bm25.fit(docs)

    query = "neural network learning"
    sparse_results = bm25.search(query, top_k=5)
    print("Sparse (BM25) results:")
    for doc, score in sparse_results:
        print(f"  {score:.3f} -> {doc[:60]}")

    dense_results = [
        (docs[1], 0.92),
        (docs[0], 0.85),
        (docs[3], 0.45),
        (docs[4], 0.30),
        (docs[2], 0.25),
    ]
    print("\nDense (embedding) results:")
    for doc, score in dense_results:
        print(f"  {score:.3f} -> {doc[:60]}")

    hybrid = HybridSearch()
    fused = hybrid.search(query, dense_results, top_k=3)
    print(f"\nHybrid (RRF-fused) results:")
    for doc, score in fused:
        print(f"  {score:.4f} -> {doc[:60]}")
    print("Hybrid search demo complete.")
