"""Pre-flight verification script for RAG From Scratch."""

import importlib
import os
import sys


def check_python():
    print(f"Python: {sys.version.split()[0]} (need 3.10+)")
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    print("  OK")


def check_dependencies():
    ok = True
    for pkg in ["tiktoken", "numpy", "requests", "bs4", "chromadb",
                "langchain", "langchain_openai", "langchain_community"]:
        try:
            importlib.import_module(pkg)
            print(f"  {pkg}: installed")
        except ImportError:
            print(f"  {pkg}: MISSING — run: pip install -r requirements.txt")
            ok = False
    return ok


def check_api_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if key.startswith("sk-"):
        print(f"  OPENAI_API_KEY: set ({key[:12]}...)")
        return True
    else:
        print("  OPENAI_API_KEY: not set")
        print("    Set: export OPENAI_API_KEY='sk-...'")
        return False


def check_rag_module():
    try:
        from rag import (cosine_similarity, num_tokens, reciprocal_rank_fusion,
                         get_unique_union, format_qa_pair, RouteQuery, TutorialSearch)
        print(f"  rag.py: imports OK")
        assert cosine_similarity([1, 0], [1, 0]) == 1.0
        assert num_tokens("test") > 0
        print("  rag.py: smoke tests OK")
        return True
    except Exception as e:
        print(f"  rag.py: ERROR — {e}")
        return False


def main():
    print("=== RAG Setup Check ===\n")
    print("[1/4] Python version:")
    check_python()
    print()
    print("[2/4] Dependencies:")
    deps_ok = check_dependencies()
    print()
    print("[3/4] API Key:")
    api_ok = check_api_key()
    print()
    print("[4/4] Module check:")
    mod_ok = check_rag_module()
    print()
    if deps_ok and mod_ok:
        print("Dependencies and module OK. Set OPENAI_API_KEY to run pipelines.")
    else:
        print("Some checks failed — see above.")


if __name__ == "__main__":
    main()
