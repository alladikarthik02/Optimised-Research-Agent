"""
`vector_qa` LangChain Tool
———————————————
Takes a natural-language question, performs a cosine-similarity search
over the “papers” Chroma collection, and returns the top-k passages
concatenated into one string (good for LLM context).
"""

from typing import List

from langchain_core.tools import Tool, BaseTool
from langchain_chroma import Chroma            # ← new import

from llamascholar.embeddings import get_embedder

CHROMA_DIR = ".chroma"
COLLECTION = "papers"


def _vector_query(query: str, k: int = 4) -> List[str]:
    """Search Chroma and return page text contents."""
    vectordb = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embedder(),
        persist_directory=CHROMA_DIR,
    )
    docs = vectordb.similarity_search(query, k=k)
    return [d.page_content for d in docs]


def build_rag_tool() -> BaseTool:
    """Factory that returns a ready-to-register Tool object."""
    return Tool(
        name="vector_qa",
        func=lambda q: "\n\n".join(_vector_query(q)),
        description=(
            "Use this to answer questions about any uploaded academic PDFs "
            "stored in the vector database. Give it a plain English question."
        ),
    )
