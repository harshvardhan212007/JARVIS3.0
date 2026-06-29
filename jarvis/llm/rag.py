from langchain_community.vectorstores import FAISS
from search.web import _should_search_web , _web_search 
def build_context(query: str, vector_db: FAISS) -> str:
    """
    Retrieves context from:
      1. Local FAISS vector store (always)
      2. DuckDuckGo web search    (only when query signals need for fresh data)
    Returns a single combined context string.
    """
    # ── Local RAG retrieval ───────────────────────────────────────────────────
    docs    = vector_db.similarity_search(query, k=4)
    doc_ctx = "\n\n".join(d.page_content for d in docs)

    sections: list[str] = []
    if doc_ctx.strip():
        sections.append(f"[From your indexed documents]\n{doc_ctx}")

    # ── Web search ────────────────────────────────────────────────────────────
    if _should_search_web(query):
        print("  [Searching the web…]", end=" ", flush=True)
        web = _web_search(query)
        if web:
            sections.append(f"[Live web results]\n{web}")
            print("done ✓")
        else:
            print("no results")

    return "\n\n───\n\n".join(sections)
