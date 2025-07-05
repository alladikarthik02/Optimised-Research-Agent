#!/usr/bin/env python
"""
Runs the Zero-Shot-ReAct LangGraph agent that powers the API & CLI.
"""

from __future__ import annotations
import os, sys
from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_cloudflare import ChatCloudflareWorkersAI

from llamascholar.tool_registry import get_tools
from llamascholar.rag_tool import build_rag_tool
from llamascholar.memory import get_memory

# 1️⃣  LLM
llm = ChatCloudflareWorkersAI(
    model="@cf/meta/llama-3-8b-instruct",
    account_id=os.environ["CF_ACCOUNT_ID"],
    api_token=os.environ["CF_API_TOKEN"],
    temperature=0.0,
)

# 2️⃣  Tool belt
TOOLS = get_tools() + [build_rag_tool()]

# 3️⃣  Compile graph (Redis or memory checkpoint)
agent_graph = create_react_agent(
    model=llm,
    tools=TOOLS,
    checkpointer=get_memory(),
    name="llamascholar-react",
)

# 4️⃣  Helper for CLI / API
def run_llamascholar(question: str, thread_id: str = "cli") -> str:
    """Return the assistant’s final reply (blocking call)."""
    res: dict[str, Any] = agent_graph.invoke(
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
    return res["messages"][-1].content

# 5️⃣  CLI entry-point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m llamascholar.graph_runner \"<question>\"")
    print("\n🔍  Answer:\n", run_llamascholar(" ".join(sys.argv[1:])))
