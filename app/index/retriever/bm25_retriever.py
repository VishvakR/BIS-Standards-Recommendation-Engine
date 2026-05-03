from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever


def bm25_retriever(
    vector_store: Chroma,
    top_k: int = 5,
) -> BM25Retriever:

    result = vector_store.get()
    texts = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    if not texts:
        raise ValueError(
            "The Chroma collection is empty — cannot build a BM25 index. "
            "Please ingest documents first."
        )

    retriever = BM25Retriever.from_texts(
        texts=texts,
        metadatas=metadatas,
    )
    retriever.k = top_k
    return retriever
