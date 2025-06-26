"""
Namespace package for reusable LangChain tools.

Importing from here keeps `agent_runner` clean:
    from llamascholar.tools import build_ddg_tool, build_arxiv_tool
"""
from .duckduckgo import build_ddg_tool   # noqa: F401  (re-export)
from .arxiv import build_arxiv_tool      # noqa: F401
