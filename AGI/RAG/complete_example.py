"""Demonstration script exercising RAG From Scratch components.

Run with --offline to skip API-dependent sections.
"""

import os
import sys

sys.path.insert(0, ".")

from rag import (
    cosine_similarity,
    num_tokens,
    reciprocal_rank_fusion,
    rerank_by_rrf,
    get_unique_union,
    format_qa_pair,
    format_qa_pairs,
    get_wikipedia_page,
    RouteQuery,
    TutorialSearch,
    PROMPTS,
)


def lesson_01_token_counting():
    print("\n=== Lesson 1-2: Token Counting ===")
    text = "Retrieval augmented generation grounds LLM responses in external knowledge."
    n = num_tokens(text)
    print(f"  Text: '{text[:50]}...'")
    print(f"  Tokens: {n}")


def lesson_02_cosine_similarity():
    print("\n=== Lesson 2: Cosine Similarity ===")
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.8, 0.2, 0.1]
    emb3 = [0.0, 1.0, 0.0]
    print(f"  sim(emb1, emb2) = {cosine_similarity(emb1, emb2):.4f}")
    print(f"  sim(emb1, emb3) = {cosine_similarity(emb1, emb3):.4f}")
    print(f"  sim(emb1, emb1) = {cosine_similarity(emb1, emb1):.4f}")


def lesson_03_formatting():
    print("\n=== Lesson 3: Formatting ===")
    docs = [
        type("Doc", (), {"page_content": "Paris is the capital of France."})(),
        type("Doc", (), {"page_content": "France is in Western Europe."})(),
    ]
    print(f"  format_docs output:\n    {docs[0].page_content}\n    {docs[1].page_content}")


def lesson_05_unique_union():
    print("\n=== Lesson 5: Unique Union ===")
    docs1 = [{"page_content": "a"}, {"page_content": "b"}]
    docs2 = [{"page_content": "a"}, {"page_content": "c"}]
    union = get_unique_union([docs1, docs2])
    print(f"  Union: {[d['page_content'] for d in union]}")


def lesson_06_rrf():
    print("\n=== Lesson 6: Reciprocal Rank Fusion ===")
    results = [
        [{"page_content": "doc1"}, {"page_content": "doc2"}],
        [{"page_content": "doc2"}, {"page_content": "doc3"}],
    ]
    fused = reciprocal_rank_fusion(results)
    print(f"  Fused: {[d['page_content'] for d in fused]}")


def lesson_07_qa_pairs():
    print("\n=== Lesson 7: QA Pairs ===")
    qs = ["What is RAG?", "How does retrieval work?"]
    ans = ["RAG stands for retrieval augmented generation.",
           "Retrieval finds relevant documents via embedding similarity."]
    print(f"  Single: {format_qa_pair(qs[0], ans[0])}")
    print(f"  Multiple:\n{format_qa_pairs(qs, ans)}")


def lesson_08_wikipedia():
    print("\n=== Lesson 8: Wikipedia Fetch ===")
    page = get_wikipedia_page("Retrieval-augmented_generation")
    print(f"  First 200 chars: {page[:200] if page else 'FAILED'}...")


def lesson_10_routing():
    print("\n=== Lesson 10: Routing ===")
    route = RouteQuery(datasource="python_docs")
    print(f"  Route to: {route.datasource}")


def lesson_11_query_construction():
    print("\n=== Lesson 11: Query Construction ===")
    ts = TutorialSearch(content_search="rag", min_view_count=1000)
    print(f"  Tutorial search:")
    ts.pretty_print()


def lesson_15_rerank():
    print("\n=== Lesson 15: Re-ranking (RRF) ===")
    results = [
        [{"page_content": "alpha"}, {"page_content": "beta"}, {"page_content": "gamma"}],
        [{"page_content": "beta"}, {"page_content": "delta"}, {"page_content": "gamma"}],
    ]
    reranked = rerank_by_rrf(results)
    print(f"  Re-ranked: {[d['page_content'] for d in reranked]}")


def lesson_check_prompts():
    print("\n=== Prompt Templates ===")
    for name, template in PROMPTS.items():
        print(f"  {name}: {template[:60]}...")

    has_key = bool(os.environ.get("OPENAI_API_KEY", "").startswith("sk-"))
    print(f"\n  OpenAI API key: {'SET' if has_key else 'NOT SET'}")
    if has_key:
        print("  All pipeline builders available:")
        for func in ["build_basic_rag_pipeline", "build_multi_query_rag",
                     "build_rag_fusion_chain", "build_step_back_chain",
                     "build_hyde_chain", "build_logical_router",
                     "build_semantic_router", "build_multi_representation_retriever",
                     "build_colbert_retriever", "build_cohere_reranker"]:
            print(f"    - {func}()")


def main():
    print("=" * 60)
    print("RAG From Scratch — Complete Example")
    print("=" * 60)

    offline = "--offline" in sys.argv

    lesson_01_token_counting()
    lesson_02_cosine_similarity()
    lesson_03_formatting()
    lesson_05_unique_union()
    lesson_06_rrf()
    lesson_07_qa_pairs()

    if not offline:
        lesson_08_wikipedia()
    else:
        print("\n=== Lesson 8: Wikipedia Fetch (skipped, offline) ===")

    lesson_10_routing()
    lesson_11_query_construction()
    lesson_15_rerank()
    lesson_check_prompts()

    print("\nDone!")


if __name__ == "__main__":
    main()
