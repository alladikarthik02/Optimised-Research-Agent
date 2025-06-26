"""
arXiv search tool (no langchain-community dependency).

Directly wraps the official ``arxiv`` Python client and returns
brief summaries of top papers.
"""

from __future__ import annotations

from typing import List

import arxiv
from langchain_core.tools import Tool


def _arxiv_search(query: str, limit: int = 5) -> List[str]:
    """Return up to *limit* papers (title + summary + link)."""
    papers = arxiv.Search(query, max_results=limit).results()
    return [
        f"{p.title} — {p.authors[0].name if p.authors else 'N/A'}\n"
        f"{p.summary.strip()}\n🔗 {p.entry_id}"
        for p in papers
    ]


def build_arxiv_tool(limit: int = 5) -> Tool:
    """Factory that returns a Tool exposing _arxiv_search()."""
    return Tool(
        name="arxiv_search",
        description=(
            f"Search arXiv.org for academic papers (up to {limit}). "
            "Input → keywords; Output → title, short summary, and link."
        ),
        func=lambda q: _arxiv_search(q, limit=limit),
    )
