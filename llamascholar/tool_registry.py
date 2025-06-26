"""
Central registry for all LlamaScholar tools.
"""

from langchain_core.tools import BaseTool
from llamascholar.tools.duckduckgo import build_ddg_tool
from llamascholar.tools.arxiv import build_arxiv_tool


def get_tools() -> list[BaseTool]:
    """Return the agent’s tool list."""
    return [
        build_ddg_tool(),
        build_arxiv_tool(),
    ]
