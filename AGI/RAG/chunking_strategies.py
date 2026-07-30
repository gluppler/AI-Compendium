"""
Chunking Strategies for RAG

Splitting documents into manageable chunks for retrieval-augmented
generation. Includes fixed-size, recursive, and semantic chunking.

https://www.pinecone.io/learn/chunking-strategies/
"""

from __future__ import annotations

import re


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks


class RecursiveChunker:
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str) -> list[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text.split()) <= self.max_size or not separators:
            return [text] if text.strip() else []

        sep = separators[0]
        if sep == "":
            words = text.split()
            return [" ".join(words[i:i + self.max_size]) for i in range(0, len(words), self.max_size)]

        parts = text.split(sep)
        result = []
        for part in parts:
            if len(part.split()) > self.max_size:
                result.extend(self._recursive_split(part, separators[1:]))
            elif part.strip():
                result.append(part.strip())
        return result


class SemanticChunker:
    def __init__(self, min_sentences: int = 2, max_sentences: int = 8):
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def chunk(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        for i in range(0, len(sentences), self.max_sentences):
            chunk = " ".join(sentences[i:i + self.max_sentences])
            chunks.append(chunk)
        return chunks


def chunk_statistics(chunks: list[str]) -> dict:
    sizes = [len(c.split()) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "avg_size": sum(sizes) / len(sizes) if sizes else 0,
    }


if __name__ == "__main__":
    text = "RAG systems combine retrieval with generation. " * 30

    fixed = FixedSizeChunker(chunk_size=30, overlap=5).chunk(text)
    print(f"Fixed-size chunks: {len(fixed)} chunks, sizes: "
          f"{[len(c.split()) for c in fixed[:3]]}...")

    recursive = RecursiveChunker(max_size=30).chunk(text)
    print(f"Recursive chunks: {len(recursive)}")

    doc = ("This is the first sentence about AI. Here is another important point. "
           "And a third sentence that follows logically. "
           "Now we change topic entirely. This is about something different. "
           "More details on the new topic. And even more. Still on this topic. "
           "One final sentence to wrap things up.")
    semantic = SemanticChunker(min_sentences=2, max_sentences=3).chunk(doc)
    print(f"\nSemantic chunks ({len(semantic)}):")
    for i, c in enumerate(semantic):
        print(f"  Chunk {i + 1}: {c[:80]}...")

    stats = chunk_statistics(semantic)
    print(f"\nChunk statistics: {stats}")
    print("Chunking strategies demo complete.")
