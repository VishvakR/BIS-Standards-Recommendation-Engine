"""
Rerank retriever.

Combines vector similarity search and BM25 keyword search via an
EnsembleRetriever, then applies a cross-encoder reranker to produce
the final, semantically-refined result set.

Pipeline:
  EnsembleRetriever(vector + BM25)
      → ContextualCompressionRetriever(CrossEncoderReranker)
"""

from langchain_chroma import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from .auto_retriever import auto_retriever
from .bm25_retriever import bm25_retriever


def rerank_retriever(
    vector_store: Chroma,
    top_k: int = 5,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ensemble_weights: tuple = (0.5, 0.5),
) -> ContextualCompressionRetriever:
    """
    Create a hybrid ensemble + cross-encoder rerank retriever.

    The candidate pool fetched by the ensemble is intentionally larger than
    top_k so the cross-encoder has enough documents to rescore before
    trimming to the final top_k.

    Args:
        vector_store:      A LangChain Chroma vector store instance.
        top_k:             Final number of documents to return after reranking.
        rerank_model:      HuggingFace cross-encoder model for reranking.
        ensemble_weights:  (vector_weight, bm25_weight) — must sum to 1.0.

    Returns:
        A ContextualCompressionRetriever that reranks ensemble results.
    """
    # Wider candidate pool so the reranker has enough to work with.
    candidate_k = max(top_k * 3, 10)

    vector_ret = auto_retriever(vector_store, top_k=candidate_k)
    bm25_ret = bm25_retriever(vector_store, top_k=candidate_k)

    ensemble = EnsembleRetriever(
        retrievers=[vector_ret, bm25_ret],
        weights=list(ensemble_weights),
    )

    cross_encoder = HuggingFaceCrossEncoder(model_name=rerank_model)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_k)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble,
    )
