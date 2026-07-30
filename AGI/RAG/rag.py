"""RAG From Scratch — 18 lessons consolidated.

Covers indexing, retrieval, generation, multi-query, RAG-fusion,
decomposition, step-back, HyDE, routing, query construction,
multi-representation indexing, ColBERT, re-ranking, CRAG, Self-RAG,
and long-context impact.

Requires: pip install langchain langchain-community langchain-openai
          chromadb tiktoken beautifulsoup4 numpy
Optional: ragatouille cohere langgraph
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

# ── Lesson 2: Token Counting ────────────────────────────────────────────

def num_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

# ── Lesson 2: Cosine Similarity ─────────────────────────────────────────

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = math.sqrt(sum(a * a for a in vec1))
    n2 = math.sqrt(sum(b * b for b in vec2))
    return dot / (n1 * n2) if n1 * n2 > 0 else 0.0

# ── Lesson 3: Formatting ────────────────────────────────────────────────

def format_docs(docs: list[Any]) -> str:
    return "\n\n".join(d.page_content for d in docs)

# ── Lesson 5: Unique Union ──────────────────────────────────────────────

def get_unique_union(documents: list[list[Any]]) -> list[Any]:
    seen = set()
    result = []
    for sublist in documents:
        for doc in sublist:
            key = doc.page_content if hasattr(doc, "page_content") else str(doc)
            if key not in seen:
                seen.add(key)
                result.append(doc)
    return result

# ── Lesson 6: Reciprocal Rank Fusion ────────────────────────────────────

def reciprocal_rank_fusion(results: list[list[Any]], k: int = 60) -> list[Any]:
    fused: dict[str, tuple[float, Any]] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = doc.page_content if hasattr(doc, "page_content") else str(doc)
            score, _ = fused.get(key, (0.0, doc))
            fused[key] = (score + 1.0 / (rank + k), doc)
    ranked = sorted(fused.items(), key=lambda x: x[1][0], reverse=True)
    return [doc for _, (_, doc) in ranked]

# ── Lesson 7: QA Pair Formatting ────────────────────────────────────────

def format_qa_pair(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"

def format_qa_pairs(questions: list[str], answers: list[str]) -> str:
    parts = []
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        parts.append(f"Question {i}: {q}\nAnswer {i}: {a}")
    return "\n\n".join(parts)

# ── Lesson 10: Route Query Schema ───────────────────────────────────────

class RouteQuery:
    def __init__(self, datasource: str):
        self.datasource = datasource

# ── Lesson 11: Tutorial Search Schema ───────────────────────────────────

class TutorialSearch:
    def __init__(self, content_search: str = "", title_search: str = "",
                 min_view_count: int | None = None,
                 max_view_count: int | None = None,
                 earliest_publish_date: str | None = None,
                 latest_publish_date: str | None = None,
                 min_length_sec: int | None = None,
                 max_length_sec: int | None = None):
        self.content_search = content_search
        self.title_search = title_search
        self.min_view_count = min_view_count
        self.max_view_count = max_view_count
        self.earliest_publish_date = earliest_publish_date
        self.latest_publish_date = latest_publish_date
        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec

    def pretty_print(self):
        for k, v in vars(self).items():
            if v is not None:
                print(f"{k}: {v}")

# ── Lesson 12: Wikipedia Page Fetch ─────────────────────────────────────

def get_wikipedia_page(title: str) -> str | None:
    import requests
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json",
        "titles": title, "prop": "extracts", "explaintext": True,
    }
    headers = {"User-Agent": "RAG-From-Scratch/1.0"}
    data = requests.get(url, params=params, headers=headers).json()
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract")

# ── Lesson 15: Reciprocal Rank Fusion (as re-ranker) ────────────────────

def rerank_by_rrf(initial_results: list[list[Any]], k: int = 60) -> list[Any]:
    return reciprocal_rank_fusion(initial_results, k)

# =========================================================================
# Prompt templates for all 18 lessons
# =========================================================================

PROMPTS: dict[str, str] = {
    "rag": "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}",
    "multi_query": "You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.\nOriginal question: {question}",
    "rag_fusion": "You are a helpful assistant that generates multiple search queries based on a single input query.\nGenerate multiple search queries related to: {question}\nOutput (4 queries):",
    "decomposition": "You are a helpful assistant that generates multiple sub-questions related to an input question.\nThe goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation.\nGenerate multiple search queries related to: {question}\nOutput (3 queries):",
    "decomposition_rag": "Here is the question you need to answer:\n\n --- \n{question}\n --- \n\nHere is any available background question + answer pairs:\n\n --- \n{q_a_pairs}\n --- \n\nHere is additional context relevant to the question:\n\n --- \n{context}\n --- \n\nUse the above context and any background question + answer pairs to answer the question:\n{question}",
    "step_back_system": "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.",
    "step_back_response": "You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.\n\n# {normal_context}\n# {step_back_context}\n\n# Original Question: {question}\n# Answer:",
    "hyde": "Please write a scientific paper passage to answer the question\nQuestion: {question}\nPassage:",
    "route_system": "You are an expert at routing a user question to the appropriate data source. Based on the programming language the question is referring to, route it to the relevant data source.",
    "query_analysis_system": "You are an expert at converting user questions into database queries. You have access to a database of tutorial videos about a software library for building LLM-powered applications. Given a question, return a database query optimized to retrieve the most relevant results. If there are acronyms or words you are not familiar with, do not try to rephrase them.",
    "summarize": "Summarize the following document:\n\n{doc}",
    "synthesize": "Here is a set of Q+A pairs:\n\n{context}\n\nUse these to synthesize an answer to the question: {question}",
}

# =========================================================================
# High-level RAG pipeline builder (lessons 1-4)
# =========================================================================

def build_basic_rag_pipeline(
    web_url: str = "https://lilianweng.github.io/posts/2023-06-23-agent/",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 4,
) -> dict[str, Any]:
    """Build a basic RAG chain using LangChain components.

    Requires: OPENAI_API_KEY env var, langchain libraries installed.
    Returns a dict with retriever, llm, prompt, and chain.
    """
    from langchain import hub
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    import bs4

    loader = WebBaseLoader(
        web_paths=(web_url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header"))),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return {"retriever": retriever, "llm": llm, "prompt": prompt, "chain": chain}

# =========================================================================
# Multi-Query RAG (lesson 5)
# =========================================================================

def build_multi_query_rag(retriever: Any) -> Any:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from operator import itemgetter

    prompt = ChatPromptTemplate.from_template(PROMPTS["multi_query"])
    generate_queries = (
        prompt | ChatOpenAI(temperature=0) | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    retrieval_chain = generate_queries | retriever.map() | get_unique_union

    rag_prompt = ChatPromptTemplate.from_template(PROMPTS["rag"])
    llm = ChatOpenAI(temperature=0)

    return (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | rag_prompt | llm | StrOutputParser()
    )

# =========================================================================
# RAG-Fusion (lesson 6)
# =========================================================================

def build_rag_fusion_chain(retriever: Any) -> Any:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from operator import itemgetter

    prompt = ChatPromptTemplate.from_template(PROMPTS["rag_fusion"])
    generate_queries = (
        prompt | ChatOpenAI(temperature=0) | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    retrieval_chain = (
        generate_queries | retriever.map() | reciprocal_rank_fusion
    )
    rag_prompt = ChatPromptTemplate.from_template(PROMPTS["rag"])
    llm = ChatOpenAI(temperature=0)

    return (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | rag_prompt | llm | StrOutputParser()
    )

# =========================================================================
# Decomposition (lesson 7)
# =========================================================================

def decomposition_retrieve_and_rag(
    question: str, retriever: Any, llm: Any, prompt_rag: Any
) -> tuple[list[str], list[str]]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts import ChatPromptTemplate
    from operator import itemgetter

    decomp_prompt = ChatPromptTemplate.from_template(PROMPTS["decomposition"])
    sub_q_gen = decomp_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    sub_questions = sub_q_gen.invoke({"question": question})

    rag_results = []
    for sq in sub_questions:
        docs = retriever.get_relevant_documents(sq)
        ans = (prompt_rag | llm | StrOutputParser()).invoke(
            {"context": docs, "question": sq})
        rag_results.append(ans)

    return sub_questions, rag_results

def answer_by_decomposition(
    question: str, retriever: Any, llm: Any,
    strategy: str = "recursive",
) -> str:
    """Answer by decomposition using recursive or individual strategy."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts import ChatPromptTemplate
    from operator import itemgetter

    decomp_prompt = ChatPromptTemplate.from_template(PROMPTS["decomposition"])
    sub_q_gen = decomp_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    sub_questions = sub_q_gen.invoke({"question": question})

    if strategy == "recursive":
        q_a_pairs = ""
        for q in sub_questions:
            prompt = ChatPromptTemplate.from_template(PROMPTS["decomposition_rag"])
            chain = (
                {"context": itemgetter("question") | retriever,
                 "question": itemgetter("question"),
                 "q_a_pairs": itemgetter("q_a_pairs")}
                | prompt | llm | StrOutputParser()
            )
            answer = chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
            q_a_pairs += "\n---\n" + format_qa_pair(q, answer)
        return q_a_pairs
    else:
        questions, answers = decomposition_retrieve_and_rag(question, retriever, llm, None)
        context = format_qa_pairs(questions, answers)
        prompt = ChatPromptTemplate.from_template(PROMPTS["synthesize"])
        return (prompt | llm | StrOutputParser()).invoke(
            {"context": context, "question": question})

# =========================================================================
# Step-Back (lesson 8)
# =========================================================================

def build_step_back_chain(retriever: Any) -> Any:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

    examples = [
        {"input": "Could the members of The Police perform lawful arrests?",
         "output": "what can the members of The Police do?"},
        {"input": "Jan Sindel's was born in what country?",
         "output": "what is Jan Sindel's personal history?"},
    ]
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"), ("ai", "{output}")])
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt, examples=examples)
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPTS["step_back_system"] + " Here are a few examples:"),
        few_shot, ("user", "{question}"),
    ])
    step_back_gen = prompt | ChatOpenAI(temperature=0) | StrOutputParser()

    response_prompt = ChatPromptTemplate.from_template(PROMPTS["step_back_response"])
    return (
        {
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            "step_back_context": step_back_gen | retriever,
            "question": lambda x: x["question"],
        }
        | response_prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    )

# =========================================================================
# HyDE (lesson 9)
# =========================================================================

def build_hyde_chain(retriever: Any) -> Any:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(PROMPTS["hyde"])
    hyde_chain = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    retrieval_chain = hyde_chain | retriever
    rag_prompt = ChatPromptTemplate.from_template(PROMPTS["rag"])
    llm = ChatOpenAI(temperature=0)

    return (
        {"context": retrieval_chain, "question": lambda x: x["question"]}
        | rag_prompt | llm
    )

# =========================================================================
# Logical Routing (lesson 10a)
# =========================================================================

def build_logical_router() -> Any:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI
    from typing import Literal
    from langchain_core.pydantic_v1 import BaseModel, Field

    class _RouteQuery(BaseModel):
        datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
            ..., description="Route user question to the most relevant datasource")

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured = llm.with_structured_output(_RouteQuery)
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPTS["route_system"]),
        ("human", "{question}"),
    ])
    router = prompt | structured

    def _choose(result):
        ds = result.datasource.lower()
        if "python" in ds:
            return "chain for python_docs"
        elif "js" in ds:
            return "chain for js_docs"
        return "chain for golang_docs"

    return router | RunnableLambda(_choose)

# =========================================================================
# Semantic Routing (lesson 10b)
# =========================================================================

def build_semantic_router() -> Any:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_core.prompts import PromptTemplate

    physics = "You are a very smart physics professor...\nHere is a question:\n{query}"
    math = "You are a very good mathematician...\nHere is a question:\n{query}"
    templates = [physics, math]
    emb = OpenAIEmbeddings()
    prompt_embeddings = emb.embed_documents(templates)

    def _router(input):
        q_emb = emb.embed_query(input["query"])
        sims = [cosine_similarity(q_emb, pe) for pe in prompt_embeddings]
        idx = sims.index(max(sims))
        print(f"Using {'MATH' if idx == 1 else 'PHYSICS'}")
        return PromptTemplate.from_template(templates[idx])

    return (
        {"query": RunnablePassthrough()}
        | RunnableLambda(_router) | ChatOpenAI() | StrOutputParser()
    )

# =========================================================================
# Multi-representation Indexing (lesson 12)
# =========================================================================

def build_multi_representation_retriever(
    urls: list[str],
) -> Any:
    """Build a MultiVectorRetriever that stores summaries in vectorstore
    and full documents in a byte store."""
    import uuid
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.storage import InMemoryByteStore
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain.retrievers.multi_vector import MultiVectorRetriever

    docs = []
    for url in urls:
        docs.extend(WebBaseLoader(url).load())

    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template(PROMPTS["summarize"])
        | ChatOpenAI(model="gpt-3.5-turbo", max_retries=0)
        | StrOutputParser()
    )
    summaries = chain.batch(docs, {"max_concurrency": 5})

    vectorstore = Chroma(collection_name="summaries",
                         embedding_function=OpenAIEmbeddings())
    store = InMemoryByteStore()
    id_key = "doc_id"
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]
    vectorstore.add_documents(summary_docs)
    store.mset(list(zip(doc_ids, docs)))

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, byte_store=store, id_key=id_key)
    return retriever

# =========================================================================
# ColBERT via RAGatouille (lesson 14)
# =========================================================================

def build_colbert_retriever(
    document: str = "",
    index_name: str = "my_index",
) -> Any:
    """Build a ColBERT retriever using RAGatouille.

    Requires: pip install ragatouille
    """
    from ragatouille import RAGPretrainedModel

    rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    if document:
        rag.index(
            collection=[document],
            index_name=index_name,
            max_document_length=180,
            split_documents=True,
        )
    return rag.as_langchain_retriever(k=3)

# =========================================================================
# Cohere Re-Rank (lesson 15)
# =========================================================================

def build_cohere_reranker(retriever: Any) -> Any:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CohereRerank

    compressor = CohereRerank()
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever)

# =========================================================================
# Demo / CLI
# =========================================================================

def quick_demo():
    """Run all import-level tests to verify the module is sound."""
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([], []) == 0.0
    print("cosine_similarity: OK")

    assert num_tokens("hello world") > 0
    print("num_tokens: OK")

    rrf = reciprocal_rank_fusion(
        [[{"page_content": "a"}, {"page_content": "b"}]], k=60)
    assert len(rrf) == 2
    print("reciprocal_rank_fusion: OK")

    assert len(get_unique_union([[{"page_content": "a"}, {"page_content": "b"}],
                              [{"page_content": "a"}]])
    ) == 2
    print("get_unique_union: OK")

    qa = format_qa_pair("q1", "a1")
    assert "q1" in qa and "a1" in qa
    print("format_qa_pair: OK")

    page = get_wikipedia_page("Python_(programming_language)")
    assert page is not None and len(page) > 100
    print("get_wikipedia_page: OK")

    rs = RouteQuery("python_docs")
    assert rs.datasource == "python_docs"
    print("RouteQuery: OK")

    ts = TutorialSearch(content_search="rag")
    assert ts.content_search == "rag"
    print("TutorialSearch: OK")

    assert PROMPTS["rag"] is not None
    assert PROMPTS["multi_query"] is not None
    assert len(PROMPTS) == 12
    print("All prompts loaded: OK")

    print("\nAll quick demo tests passed!")


def main():
    import sys
    if "--demo" in sys.argv:
        quick_demo()
    else:
        print("RAG From Scratch — 18 lessons consolidated.")
        print("Usage: python rag.py --demo")
        print("\nTo build RAG pipelines, set OPENAI_API_KEY and run:")
        print("  from rag import build_basic_rag_pipeline")
        print("  chain = build_basic_rag_pipeline()")
        print("  chain.invoke('What is task decomposition?')")


if __name__ == "__main__":
    main()
