"""
Auto (vector-similarity) retriever.

Returns a LangChain VectorStoreRetriever backed by the Chroma
vector store.  This is the default strategy when no explicit
query-engine type is requested.
"""

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever


def auto_retriever(
    vector_store: Chroma,
    top_k: int = 5,
) -> VectorStoreRetriever:
    """
    Create a standard vector-similarity retriever.

    Args:
        vector_store: A LangChain Chroma vector store instance.
        top_k:        Number of documents to retrieve per query.

    Returns:
        A VectorStoreRetriever configured for similarity search.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
