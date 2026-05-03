import os
import logging
from typing import Optional, List

from langchain_core.documents import Document

from app.llm.main import LanguageModel
from app.llm.manager import LLMManager
from app.storage.manager import StoreManager
from app.index.main import IndexBase
from app.index.manager import IndexManager
from app.ingestion.parsing import Documents

logger = logging.getLogger(__name__)


class RAGAgent:

    def __init__(
        self,
        llm: LanguageModel,
        storage: StoreManager,
        index: IndexBase,
        document: Optional[Documents] = None,
        phoenix_api_key: Optional[str] = None,
    ):
        if llm is None:
            raise TypeError("Missing required argument 'llm'")
        if storage is None:
            raise TypeError("Missing required argument 'storage'")
        if index is None:
            raise TypeError("Missing required argument 'index'")

        self.llm = llm
        self.storage_manager = storage
        self.index_manager = index
        self.document_handler = document

        # Optional: enable Arize Phoenix tracing via OpenTelemetry.
        # Set the OTLP auth header so traces are streamed to llamatrace.com.
        if phoenix_api_key:
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={phoenix_api_key}"
            logger.info("Arize Phoenix tracing enabled.")


    def create_index(self) -> None:
        """Ingest documents and build (or refresh) the vector index."""
        self.index_manager.create_index()

 
    def get_relevant_nodes(self, prompt: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve the most relevant document chunks for *prompt*.
        """
        return self.index_manager.query(prompt)

    def get_documents(self) -> List[Document]:
        """
        Load and return the raw source documents (before chunking/indexing).
        """
        if self.document_handler is None:
            raise ValueError(
                "No document handler set. Pass a Documents instance when "
                "constructing RAGAgent."
            )
        return self.document_handler.load_documents()

 
    def run(self, prompt: str) -> str:
        """
        Full RAG pipeline: retrieve relevant chunks then generate a response.

        Args:
            prompt: The user query.

        Returns:
            A generated answer string from the LLM.
        """
        docs = self.get_relevant_nodes(prompt)

        context = "\n\n".join(doc.page_content for doc in docs)

        augmented_prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}"
        )

        return self.llm.generate_response(augmented_prompt)


# ═══════════════════════════════════════════════════════
# BIS Standards Recommendation Engine
# ═══════════════════════════════════════════════════════

from app.models.schemas import StandardRecommendation, BISResult
from app.index.retriever.hybrid_bis_retriever import HybridBISRetriever


class BISRecommendationEngine:
    """
    Main pipeline for the BIS Standards Recommendation Engine.

    Initialises the hybrid retriever once and exposes a `.recommend()` method
    that inference.py calls per query.  No LLM is loaded by default to keep
    latency low — rationale generation is only used in the API / UI.
    """

    def __init__(
        self,
        collection_name: str = "bis_standards",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        persist_dir: str = "./storage/chromadb",
        use_llm: bool = False,
        llm_model_type: str = "opensource",
        llm_model_name: str = "llama3.1:8b",
    ):
        logger.info("Initializing BISRecommendationEngine …")

        self.retriever = HybridBISRetriever(
            collection_name=collection_name,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
            persist_dir=persist_dir,
        )

        # Optional LLM for rationale generation (API / UI only)
        self.rationale_gen = None
        if use_llm:
            try:
                from app.llm.rationale_generator import RationaleGenerator
                llm = LLMManager.create(
                    model_type=llm_model_type,
                    model_name=llm_model_name,
                )
                self.rationale_gen = RationaleGenerator(llm)
                logger.info("LLM rationale generator loaded.")
            except Exception as e:
                logger.warning(f"Failed to load LLM for rationale: {e}")

        logger.info("BISRecommendationEngine ready.")

    def recommend(
        self,
        query: str,
        top_k: int = 5,
        generate_rationale: bool = False,
    ) -> List[StandardRecommendation]:
        """
        Recommend BIS standards for a product description.

        Args:
            query:              Product description string.
            top_k:              Maximum number of standards to return.
            generate_rationale: If True and LLM is loaded, generate rationale.

        Returns:
            List of StandardRecommendation (max top_k), ordered by relevance.
        """
        # Step 1 — Hybrid retrieval
        results: List[BISResult] = self.retriever.retrieve(query, top_k=top_k)

        # Step 2 — Optional rationale generation
        if generate_rationale and self.rationale_gen is not None:
            try:
                results = self.rationale_gen.generate(query, results)
            except Exception as e:
                logger.warning(f"Rationale generation failed: {e}")

        # Step 3 — Convert to Pydantic models
        recommendations = []
        for r in results:
            recommendations.append(
                StandardRecommendation(
                    standard_id=r.standard_id,
                    title=r.title,
                    category=r.category,
                    scope=r.scope,
                    rationale=getattr(r, "rationale", ""),
                    relevance_score=r.relevance_score,
                )
            )

        return recommendations[:top_k]