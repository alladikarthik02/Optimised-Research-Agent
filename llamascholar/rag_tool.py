"""
vector_qa tool: takes a user question, performs cosine-similarity search
on the Chroma store, and returns top-k context passages as a single
string (LLM-friendly).
"""

from typing import Any, List

from langchain_core.tools import BaseTool, Tool
from langchain_community.vectorstores import Chroma

from llamascholar.embeddings import get_embedder

CHROMA_DIR = ".chroma"
COLLECTION = "papers"


def _vector_query(query: str, k: int = 4) -> List[str]:
    vectordb = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embedder(),
        persist_directory=CHROMA_DIR,
    )
    docs = vectordb.similarity_search(query, k=k)
    return [d.page_content for d in docs]


def build_rag_tool() -> BaseTool:
    return Tool(
        name="vector_qa",
        func=lambda q: "\n\n".join(_vector_query(q)),
        description=(
            "Answer questions about uploaded academic PDFs. "
            "Input: natural-language question. "
            "Output: relevant passages (plain text). "
            "Use this after searching arXiv when deeper detail is required."
        ),
    )
