#!/usr/bin/env python
"""
llamascholar/graph_runner.py
────────────────────────────────────────────────────────────
• Zero-Shot ReAct agent powered by Cloudflare Workers-AI
  (@cf/meta/llama-3-8b-instruct)
• Tool belt: duckduckgo_search, arxiv_search, vector_qa (Chroma RAG)
• Chat history stored in Redis if REDIS_URL is available; otherwise
  in-process memory.
• Usable both as a CLI script and as an import for FastAPI.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_cloudflare import ChatCloudflareWorkersAI

from llamascholar.tool_registry import get_tools
from llamascholar.rag_tool import build_rag_tool
from llamascholar.memory import get_memory           # <- returns RedisSaver() or InMemorySaver()

# ───────────────────────── 1.  LLM ───────────────────────── #
llm = ChatCloudflareWorkersAI(
    model="@cf/meta/llama-3-8b-instruct",          # full alias incl. @cf/
    account_id=os.environ["CF_ACCOUNT_ID"],
    api_token=os.environ["CF_API_TOKEN"],
    temperature=0.0,
    streaming=True,
)

# ───────────────────────── 2.  Tools ─────────────────────── #
TOOLS = get_tools() + [build_rag_tool()]

# ───────────────────────── 3.  Agent graph ───────────────── #
agent_graph = create_react_agent(
    model=llm,
    tools=TOOLS,
    checkpointer=get_memory(),                     # Redis or fallback
    name="llamascholar-react",
)  # default prompt comes from LangGraph

# ───────────────────────── 4.  CLI helper ────────────────── #
def ask(question: str, thread_id: str = "cli") -> None:
    """
    Send *question* to the agent and print the final assistant reply.
    `thread_id` lets you keep multi-turn context.
    """
    result: dict[str, Any] = agent_graph.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are LlamaScholar, a concise research assistant. "
                        "Think step-by-step and cite sources when helpful."
                    ),
                },
                {"role": "user", "content": question},
            ]
        },
        {"configurable": {"thread_id": thread_id}},
    )
    print("\n🔍  Answer:\n", result["messages"][-1].content)


# ───────────────────────── 5.  Entrypoint ────────────────── #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m llamascholar.graph_runner \"<your question>\"")
    ask(" ".join(sys.argv[1:]))
