"""
LangGraph DAG   (Day-3 minimal)
user_input --> router --> (vector_qa | react_agent) --> END
"""

import os
import time
from typing import Dict

import wandb
from langchain.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.cloudflare_workersai import (
    ChatCloudflareWorkersAI,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import LLMNode, ToolNode

from llamascholar.tool_registry import get_tools
from llamascholar.rag_tool import build_rag_tool

# ─────────────────── WandB init ──────────────────── #
wandb.init(project="llamascholar", entity="<your-user>", mode="online")

# ─────────────────── Build LLM ────────────────────── #
LLM = ChatCloudflareWorkersAI(
    model="meta/llama-3-8b-instruct",
    cloudflare_account_id=os.environ["CF_ACCOUNT_ID"],
    cloudflare_api_token=os.environ["CF_API_TOKEN"],
    temperature=0.0,
)

# ─────────────────── Tools & Agent ────────────────── #
TOOLS = get_tools() + [build_rag_tool()]

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are LlamaScholar, a concise research assistant."),
        ("user", "{input}"),
        ("system", "{tools}"),
        ("agent_scratchpad", "{agent_scratchpad}"),
    ]
)

react_agent = create_react_agent(LLM, TOOLS, PROMPT)
react_node = LLMNode(react_agent)

vector_node = ToolNode([build_rag_tool()])  # wraps vector_qa
# (No explicit memory node yet—will add real memory Day-4)

# ─────────────────── Router function ──────────────── #
def route(state: Dict) -> str:
    q = state["input"].lower()
    return "vector" if any(tok in q for tok in ("pdf", "paper", "section")) else "react"

# ─────────────────── Build graph ──────────────────── #
wf = StateGraph()
wf.add_node("react", react_node)
wf.add_node("vector", vector_node)

wf.set_entry_point("router").set_router(route)
wf.add_edge("react", END)
wf.add_edge("vector", END)

graph = wf.compile()


# ─────────────────── CLI runner ───────────────────── #
def run(question: str) -> None:
    start = time.time()
    result = graph.invoke({"input": question})
    wandb.log({"latency_ms": (time.time() - start) * 1000})
    print("\n🔍 Answer:\n", result["output"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("Usage: python -m llamascholar.graph_runner \"<your question>\"")
    run(sys.argv[1])
