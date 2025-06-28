"""
Shared local embedder.

Compatible with LangChain ≥ 0.2.  Requires:
    poetry add langchain_huggingface  # lightweight wheel (~13 kB)
"""


from langchain_huggingface import HuggingFaceEmbeddings




def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # show_progress_bar=False   ← REMOVE (no longer accepted)
    )
