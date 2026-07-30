"""
Context Integration

Assembles retrieved document chunks into a coherent prompt for
the generation step in RAG. Handles formatting, ordering, and
token budget management.

https://arxiv.org/abs/2005.11401
"""

from __future__ import annotations


class ContextIntegrator:
    def __init__(self, max_context_tokens: int = 2000):
        self.max_tokens = max_context_tokens

    def integrate(self, query: str, documents: list[str], scores: list[float] | None = None) -> str:
        context_parts = []
        used_tokens = 0
        for i, doc in enumerate(documents):
            doc_tokens = len(doc.split())
            if used_tokens + doc_tokens > self.max_tokens:
                remaining = self.max_tokens - used_tokens
                if remaining > 10:
                    truncated = " ".join(doc.split()[:remaining])
                    score_str = f" [Score: {scores[i]:.3f}]" if scores else ""
                    context_parts.append(f"[Document {i + 1}]{score_str}\n{truncated}")
                break
            score_str = f" [Score: {scores[i]:.3f}]" if scores else ""
            context_parts.append(f"[Document {i + 1}]{score_str}\n{doc}")
            used_tokens += doc_tokens

        context = "\n\n".join(context_parts)
        return (
            f"Use the following documents to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )


class StreamingContextIntegrator(ContextIntegrator):
    def integrate(self, query: str, documents: list[str], scores: list[float] | None = None) -> str:
        sorted_docs = sorted(
            enumerate(documents),
            key=lambda x: scores[x[0]] if scores else 0,
            reverse=True,
        )
        doc_list = [doc for _, doc in sorted_docs]
        doc_scores = [scores[i] for i, _ in sorted_docs] if scores else None
        return super().integrate(query, doc_list, doc_scores)


def format_with_citations(query: str, documents: list[tuple[str, int]]) -> str:
    citations = []
    for doc_text, doc_id in documents:
        citations.append(f"[{doc_id}] {doc_text[:100]}")
    context = "\n".join(citations)
    return (
        f"Context with citations:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer (cite sources using [n] notation):"
    )


if __name__ == "__main__":
    integrator = ContextIntegrator(max_context_tokens=30)
    query = "What are the benefits of RAG?"
    docs = [
        "RAG combines retrieval with generation for better accuracy.",
        "External knowledge sources reduce hallucination in LLMs.",
        "Vector databases enable efficient similarity search for RAG.",
        "RAG systems can access up-to-date information beyond training data.",
    ]
    scores = [0.95, 0.85, 0.72, 0.60]

    prompt = integrator.integrate(query, docs, scores)
    print("=== Integrated RAG prompt ===")
    print(prompt)

    streaming = StreamingContextIntegrator(max_context_tokens=30)
    prompt2 = streaming.integrate(query, docs, scores)
    print(f"\n=== Streaming (sorted) version ===")

    citation_prompt = format_with_citations(query, [(docs[i], i + 1) for i in range(len(docs))])
    print(f"\n=== Citation-style prompt ===")
    print(citation_prompt[:200] + "...")
    print("Context integration demo complete.")
