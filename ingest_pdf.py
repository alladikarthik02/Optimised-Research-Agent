#!/usr/bin/env python
"""
Ingest a PDF into the local Chroma vector store.

Usage:
    poetry run python ingest_pdf.py /path/to/paper.pdf
"""

import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma        # ← new import
from pypdf import PdfReader

from llamascholar.embeddings import get_embedder

CHROMA_DIR = ".chroma"      # auto-created, stays out of git
COLLECTION = "papers"       # name you’ll query from RAG


def load_text(pdf_path: str) -> str:
    """Read every page and concatenate into one big string."""
    pdf = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in pdf.pages)


def main(pdf_file: str) -> None:
    raw_text = load_text(pdf_file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,     # text length per chunk
        chunk_overlap=128,   # keeps context across boundaries
    )
    docs = splitter.create_documents([raw_text])

    vectordb = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embedder(),       # sentence-transformers
        persist_directory=CHROMA_DIR,
    )
    vectordb.add_documents(docs)    # auto-persists in 0.4.x+

    print(f"✅  Ingested {len(docs)} chunks into {CHROMA_DIR}/{COLLECTION}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        sys.exit("❌  Provide a valid PDF path")
    main(sys.argv[1])
