"""
DuckDuckGo search tool (no langchain-community dependency).

Uses the ``duckduckgo_search`` library under the hood and exposes
a LangChain-Core ``Tool``.
"""

from __future__ import annotations

from typing import List

from duckduckgo_search import DDGS
from langchain_core.tools import Tool


def _ddg_search(query: str, max_results: int = 5) -> List[str]:
    """Return up to *max_results* DuckDuckGo results (snippet + URL)."""
    results: list[str] = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=max_results):
            results.append(f"{hit['body']}  (🔗 {hit['href']})")
    return results


def build_ddg_tool(max_results: int = 5) -> Tool:
    """Factory that returns a ready-to-use Tool instance."""
    return Tool(
        name="duckduckgo_search",
        description=(
            "DuckDuckGo web search. Ideal for current events, definitions, "
            "or quick fact-checks. Input → plain-English query; "
            "Output → list of text snippets with URLs."
        ),
        func=lambda q: _ddg_search(q, max_results=max_results),
    )
