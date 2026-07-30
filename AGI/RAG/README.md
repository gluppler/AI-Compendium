# RAG From Scratch — 18 Lessons

Single-file retrieval augmented generation system consolidating LangChain's
RAG From Scratch course. No frameworks, no cloud APIs, no hidden reasoning.

## Requirements

- Python 3.10+
- OpenAI API key (for embedding + generation)

## Setup

```bash
pip install -r requirements.txt
python setup_check.py
export OPENAI_API_KEY="sk-..."
```

## Quick Start

```python
from rag import build_basic_rag_pipeline
chain = build_basic_rag_pipeline()
print(chain.invoke("What is task decomposition?"))
```

## Capabilities (18 Lessons)

| # | Capability | Function | Description |
|---|-----------|----------|-------------|
| 1-2 | Indexing | `build_basic_rag_pipeline()` | Load, split, embed, store |
| 3 | Retrieval | `format_docs()` / vectorstore | Cosine similarity + MMR |
| 4 | Generation | RAG prompt chain | Context-grounded LLM response |
| 5 | Multi-Query | `build_multi_query_rag()` | 5 perspective queries + union |
| 6 | RAG-Fusion | `build_rag_fusion_chain()` | RRF over multi-query results |
| 7 | Decomposition | `answer_by_decomposition()` | Recursive / individual Q&A |
| 8 | Step-Back | `build_step_back_chain()` | Abstract → retrieve → answer |
| 9 | HyDE | `build_hyde_chain()` | Hypothetical doc embedding |
| 10 | Routing | `build_logical_router()` / `build_semantic_router()` | LLM / embedding-based |
| 11 | Query Construction | `TutorialSearch` | Structured DB query builder |
| 12 | Multi-rep Index | `build_multi_representation_retriever()` | Summary → doc retrieval |
| 13 | RAPTOR | (see source notebooks) | Tree-based summarization |
| 14 | ColBERT | `build_colbert_retriever()` | Late interaction scoring |
| 15 | Re-ranking | `build_cohere_reranker()` / `rerank_by_rrf()` | Second-pass scoring |
| 16 | CRAG | (see source notebooks) | Corrective retrieval |
| 17 | Self-RAG | (see source notebooks) | Self-reflection |
| 18 | Long-Context | (see source notebooks) | Zero-shot vs RAG comparison |

## Structure

- `rag.py` — Single-file RAG system
- `complete_example.py` — Demo exercising pipeline components
- `setup_check.py` — Pre-flight verification
- `chunking_strategies.py` — Standalone chunking utilities
- `embedding_pipeline.py` — Standalone embedding pipeline
- `vector_search.py` — Standalone vector search
- `hybrid_search.py` — Standalone hybrid search
- `re_ranker.py` — Standalone re-ranking
- `retrieval_evaluation.py` — Standalone evaluation
- `context_integration.py` — Standalone context integration

## Source

Based on LangChain's [RAG From Scratch](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) video series.
