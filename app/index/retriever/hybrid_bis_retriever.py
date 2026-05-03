"""
Hybrid BIS Retriever — three-phase retrieval for BIS standards.

Phase 1 (RECALL):   Vector search + BM25 → top 20 each, merge & deduplicate
Phase 2 (RERANK):   Cross-encoder reranking of all candidates
Phase 3 (GROUP):    Group by standard_id, return best chunk per standard

Uses native ChromaDB client (same as ingestion) for consistency.
"""

import logging
import os
from collections import defaultdict
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from rank_bm25 import BM25Okapi

from app.models.schemas import BISResult

logger = logging.getLogger(__name__)


class HybridBISRetriever:

    def __init__(
        self,
        collection_name: str = "bis_standards",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        persist_dir: str = "./storage/chromadb",
        recall_k: int = 20,
    ):
        logger.info("Initializing HybridBISRetriever …")

        # Native ChromaDB client (same as ingestion)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.emb_fn,
        )

        # Cross-encoder for reranking
        logger.info(f"Loading cross-encoder: {rerank_model}")
        self.cross_encoder = HuggingFaceCrossEncoder(model_name=rerank_model)

        self.recall_k = recall_k

        # BM25 — built lazily
        self._bm25_index = None
        self._bm25_docs = None
        self._bm25_metas = None
        self._bm25_ids = None

        logger.info("HybridBISRetriever ready.")

    # ── BM25 index (lazy) ────────────────────────────────────────────────

    def _build_bm25_index(self):
        if self._bm25_index is not None:
            return

        logger.info("Building BM25 index from ChromaDB …")
        result = self.collection.get(include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        ids = result.get("ids", [])

        if not docs:
            logger.warning("Collection empty — BM25 skipped.")
            return

        tokenized = [d.lower().split() for d in docs]
        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_docs = docs
        self._bm25_metas = metas
        self._bm25_ids = ids
        logger.info(f"BM25 index: {len(docs)} documents.")

    # ── Phase 1: RECALL ──────────────────────────────────────────────────

    def _vector_recall(self, query: str, k: int) -> List[dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        candidates = []
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            # ChromaDB cosine distance → similarity
            score = max(0.0, 1.0 - dist)
            candidates.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "vector_score": score,
            })
        return candidates

    def _bm25_recall(self, query: str, k: int) -> List[dict]:
        self._build_bm25_index()
        if self._bm25_index is None:
            return []

        scores = self._bm25_index.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        mx = max(scores) if max(scores) > 0 else 1.0

        candidates = []
        for idx in top_idx:
            if scores[idx] > 0:
                candidates.append({
                    "text": self._bm25_docs[idx],
                    "metadata": self._bm25_metas[idx],
                    "bm25_score": scores[idx] / mx,
                })
        return candidates

    # ── Phase 2: RERANK ──────────────────────────────────────────────────

    def _rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self.cross_encoder.score(pairs)

        raw_min, raw_max = min(ce_scores), max(ce_scores)
        span = raw_max - raw_min if raw_max != raw_min else 1.0

        for i, c in enumerate(candidates):
            c["ce_score"] = (ce_scores[i] - raw_min) / span

        candidates.sort(key=lambda c: c["ce_score"], reverse=True)
        return candidates

    # ── Phase 3: GROUP by standard_id ────────────────────────────────────

    def _group_by_standard(self, candidates: List[dict], top_k: int) -> List[BISResult]:
        groups: dict = defaultdict(list)
        for c in candidates:
            sid = c.get("metadata", {}).get("standard_id", "UNKNOWN")
            groups[sid].append(c)

        results = []
        for sid, chunks in groups.items():
            best = max(chunks, key=lambda c: c.get("ce_score", 0))
            meta = best.get("metadata", {})
            results.append(BISResult(
                standard_id=sid,
                title=meta.get("title", ""),
                category=meta.get("category", ""),
                scope=meta.get("scope", ""),
                relevance_score=round(best.get("ce_score", 0.0), 4),
                context_chunk=best.get("text", ""),
            ))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    # ── Public API ───────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[BISResult]:
        # Phase 1
        vec = self._vector_recall(query, k=self.recall_k)
        bm25 = self._bm25_recall(query, k=self.recall_k)

        seen = set()
        merged = []
        for c in vec + bm25:
            key = (c["text"][:100], c["metadata"].get("standard_id", ""))
            if key not in seen:
                seen.add(key)
                merged.append(c)

        logger.info(f"Recall: {len(vec)} vec + {len(bm25)} bm25 → {len(merged)} merged")
        if not merged:
            return []

        # Phase 2
        reranked = self._rerank(query, merged)

        # Phase 3
        results = self._group_by_standard(reranked, top_k)
        logger.info(f"Returning {len(results)} standards")
        return results
