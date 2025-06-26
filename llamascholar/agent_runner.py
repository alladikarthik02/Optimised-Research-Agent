"""
Run LlamaScholar as a classic **ReAct** agent on Cloudflare Workers AI.

Tools available to the agent
    • duckduckgo_search(query: str)
    • arxiv_search(query: str)

Example (inside Poetry venv):
    poetry run python -m llamascholar.agent_runner \
        "Latest papers on open-source Llama models"
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, List

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_cloudflare import ChatCloudflareWorkersAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from llamascholar.tool_registry import get_tools

# --------------------------------------------------------------------- #
# 0.  Load CF credentials (.env can live in the repo root)              #
# --------------------------------------------------------------------- #
load_dotenv()


# --------------------------------------------------------------------- #
# 1.  LLM factory (Workers AI, Llama-3 8B)                              #
# --------------------------------------------------------------------- #
def build_llm(**kwargs: Any) -> ChatCloudflareWorkersAI:
    return ChatCloudflareWorkersAI(
        model="@cf/meta/llama-3-8b-instruct",
        account_id=os.environ["CF_ACCOUNT_ID"],
        api_token=os.environ["CF_API_TOKEN"],
        temperature=0.0,
        **kwargs,
    )


# --------------------------------------------------------------------- #
# 2.  ReAct prompt with a **minimal working example**                   #
# --------------------------------------------------------------------- #
REACT_TEMPLATE = """
You are **LlamaScholar**, a concise research assistant.
{tools}                     <-- 👈 cheat-sheet auto-filled by LangChain
(You can call: {tool_names}) <-- 👈 raw names
You have two tools:

duckduckgo_search(query) – web search, returns text snippets.
arxiv_search(query)      – search arXiv papers, returns title & summary.

**Format you MUST follow**

Thought: you should think about what to do
Action: tool_name["argument string"]
Observation: result of the action
... (the Thought/Action/Observation steps can repeat)
Thought: I now know the answer
Final Answer: <answer to user>

**Tiny example**

Thought: need a quick definition
Action: duckduckgo_search["what is a transformer model"]
Observation: A transformer is a neural-network architecture…
Thought: answer ready
Final Answer: A transformer is …

Begin!

Question: {input}
{agent_scratchpad}
""".strip()

PROMPT = PromptTemplate.from_template(REACT_TEMPLATE)


# --------------------------------------------------------------------- #
# 3.  Assemble the ReAct agent                                          #
# --------------------------------------------------------------------- #
def build_agent(tools: List[BaseTool]) -> AgentExecutor:
    llm = build_llm()
    agent = create_react_agent(llm=llm, tools=tools, prompt=PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# --------------------------------------------------------------------- #
# 4.  CLI plumbing                                                      #
# --------------------------------------------------------------------- #
async def _amain(question: str) -> None:
    executor = build_agent(get_tools())
    result = await executor.ainvoke({"input": question})
    print("\n🔍  Final answer:\n", result["output"] if isinstance(result, dict) else result)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m llamascholar.agent_runner \"<your question>\"")
        sys.exit(1)

    asyncio.run(_amain(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
