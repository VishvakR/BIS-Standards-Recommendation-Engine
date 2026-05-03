"""
Retriever strategies for AskTheSite / BIS Standards Engine.

Available retrievers:
  - auto_retriever         : Standard vector-similarity search (default).
  - bm25_retriever         : Sparse BM25 keyword search.
  - rerank_retriever       : Hybrid ensemble + cross-encoder reranking.
  - HybridBISRetriever     : Three-phase hybrid retriever for BIS standards.
"""

from .auto_retriever import auto_retriever
from .bm25_retriever import bm25_retriever
from .rerank_retriever import rerank_retriever
from .hybrid_bis_retriever import HybridBISRetriever

__all__ = ["auto_retriever", "bm25_retriever", "rerank_retriever", "HybridBISRetriever"]

