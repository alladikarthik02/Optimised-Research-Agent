#!/usr/bin/env python
"""
Usage:
    poetry run python ingest_pdf.py path/to/paper.pdf

• Extracts text → chunks with LangChain's RecursiveCharacterTextSplitter
• Embeds with all-MiniLM
• Upserts into a persistent Chroma collection "papers"
"""

import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from pypdf import PdfReader

from llamascholar.embeddings import get_embedder

CHROMA_DIR = ".chroma"
COLLECTION = "papers"


def load_text(pdf_path: str) -> str:
    pdf = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in pdf.pages)


def main(pdf_file: str) -> None:
    raw_text = load_text(pdf_file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
    )
    docs = splitter.create_documents([raw_text])

    vectordb = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embedder(),
        persist_directory=CHROMA_DIR,
    )
    vectordb.add_documents(docs)
    vectordb.persist()

    print(f"✅  Ingested {len(docs)} chunks into {CHROMA_DIR}/{COLLECTION}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        sys.exit("Provide a valid PDF path")
    main(sys.argv[1])
