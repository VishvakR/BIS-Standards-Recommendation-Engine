from typing import Optional, List

from langchain_core.documents import Document

from app.storage.main import Store
from app.index import IndexFactory
from app.ingestion import Documents
from app.index.retriever import auto_retriever, bm25_retriever, rerank_retriever
from .type import QueryEngineType


@IndexFactory.registry("chroma")
class ChromaIndex:
    def __init__(
            self,
            storage: Store,
            document: Optional[Documents] = None,
            namespace: Optional[str] = None,
            query_engine: str = QueryEngineType.AUTO.value,
            top_k: int = 5,
    ):
        self.document_handler = document
        self.storage = storage
        self.namespace = namespace
        self.query_engine = query_engine
        self.top_k = top_k

        self.index = None

    def create_index(self):
        """Create the index from the documents and persist it."""
        if self.document_handler is None:
            raise ValueError("Document handler is not provided")

        docs = self.document_handler.create_nodes()
        vector_store = self.storage.get_vector_store()
        vector_store.add_documents(docs)
        self.index = vector_store

    def _load_index(self):
        """Load the vector store from storage (lazy initialisation)."""
        vector_store = self.storage.get_vector_store()
        self.index = vector_store

    def _get_query_engine(self):
        """Return the retriever for the configured query-engine type."""
        if self.index is None:
            self._load_index()

        if self.query_engine == QueryEngineType.AUTO.value:
            return auto_retriever(self.index, top_k=self.top_k)

        elif self.query_engine == QueryEngineType.BM25.value:
            return bm25_retriever(self.index, top_k=self.top_k)

        elif self.query_engine == QueryEngineType.RERANK.value:
            return rerank_retriever(self.index, top_k=self.top_k)

        else:
            raise ValueError(
                f"Unsupported query engine type: {self.query_engine!r}. "
                f"Valid options: {[e.value for e in QueryEngineType]}"
            )

    def query(self, question: str) -> List[Document]:
        """
        Retrieve documents relevant to *question*.

        Lazily loads the vector store if not already initialised, selects
        the configured retriever strategy, and returns the top-k results.

        Args:
            question: The user query string.

        Returns:
            A list of LangChain Document objects ranked by relevance.
        """
        retriever = self._get_query_engine()
        return retriever.invoke(question)
