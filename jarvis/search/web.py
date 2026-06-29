from duckduckgo_search import DDGS

_WEB_TRIGGERS = frozenset([
    "latest", "recent", "today", "news", "right now", "current", "currently",
    "yesterday", "this week", "this month", "live", "just happened",
    "price", "stock", "weather", "score", "result", "who won",
    "2024", "2025", "2026",
])

def _should_search_web(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _WEB_TRIGGERS)


def _web_search(query: str, max_results: int = 3) -> str:
    """DuckDuckGo search — returns concatenated snippets."""
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        return "\n".join(h.get("body", "") for h in hits if h.get("body"))
    except Exception:
        return ""
