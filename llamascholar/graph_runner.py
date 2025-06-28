#!/usr/bin/env python
"""
graph_runner.py — minimal, compatible with LangGraph ≥0.3
"""
from __future__ import annotations

import os
import sys
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_cloudflare import ChatCloudflareWorkersAI

from llamascholar.tool_registry import get_tools
from llamascholar.rag_tool import build_rag_tool

# ───────── 1. LLM ───────── #
llm = ChatCloudflareWorkersAI(
    model="@cf/meta/llama-3-8b-instruct",
    account_id=os.environ["CF_ACCOUNT_ID"],
    api_token=os.environ["CF_API_TOKEN"],
    temperature=0.0,
)

# ───────── 2. Tools ─────── #
TOOLS = get_tools() + [build_rag_tool()]

# ───────── 3. Graph ─────── #
agent_graph = create_react_agent(
    model=llm,
    tools=TOOLS,
    checkpointer=InMemorySaver(),     # chat history per thread
    name="llamascholar-react",
)   # ← no prompt argument

# ───────── 4. CLI ───────── #
def ask(question: str, thread_id: str = "cli") -> None:
    """
    Send `question`, print the assistant’s final reply.
    A static system message keeps LlamaScholar’s voice.
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m llamascholar.graph_runner \"<your question>\"")
    ask(" ".join(sys.argv[1:]))
