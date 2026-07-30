"""
Embedding Pipeline

Converts text documents into vector embeddings for semantic search
and retrieval. Includes a simple embedding model implementation.

https://en.wikipedia.org/wiki/Word_embedding
"""

from __future__ import annotations

import math
import random


class EmbeddingModel:
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.dim = embedding_dim
        self.embeddings = [
            [random.gauss(0, 0.1) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        self.vocab: dict[str, int] = {}

    def fit_vocab(self, texts: list[str]) -> None:
        idx = 0
        for text in texts:
            for word in text.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1

    def embed(self, text: str) -> list[float]:
        words = text.lower().split()
        vector = [0.0] * self.dim
        count = 0
        for word in words:
            if word in self.vocab:
                idx = self.vocab[word]
                for i in range(self.dim):
                    vector[i] += self.embeddings[idx][i]
                count += 1
        if count > 0:
            norm = math.sqrt(sum(v ** 2 for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
        return vector


class EmbeddingPipeline:
    def __init__(self, embedding_dim: int = 16):
        self.model = EmbeddingModel(vocab_size=100, embedding_dim=embedding_dim)
        self.document_embeddings: list[tuple[str, list[float]]] = []
        self.fitted = False

    def fit(self, documents: list[str]) -> None:
        self.model.fit_vocab(documents)
        self.document_embeddings = [
            (doc, self.model.embed(doc)) for doc in documents
        ]
        self.fitted = True

    def embed_query(self, query: str) -> list[float]:
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted yet")
        return self.model.embed(query)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) + 1e-10
    norm_b = math.sqrt(sum(y * y for y in b)) + 1e-10
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    random.seed(42)
    docs = [
        "machine learning is transforming artificial intelligence",
        "deep neural networks learn hierarchical representations",
        "retrieval augmented generation combines search with generation",
        "natural language processing enables text understanding",
        "reinforcement learning trains agents through interaction",
    ]

    pipeline = EmbeddingPipeline(embedding_dim=16)
    pipeline.fit(docs)

    queries = ["neural networks", "search and retrieval", "AI and ML"]
    for query in queries:
        query_vec = pipeline.embed_query(query)
        similarities = [
            (doc, cosine_similarity(query_vec, doc_vec))
            for doc, doc_vec in pipeline.document_embeddings
        ]
        similarities.sort(key=lambda x: -x[1])
        print(f"\nQuery: '{query}'")
        for doc, sim in similarities:
            print(f"  {sim:.3f} -> {doc}")

    print("\nEmbedding dimension:", pipeline.model.dim)
    print("Vocabulary size:", len(pipeline.model.vocab))
    print("Embedding pipeline demo complete.")
