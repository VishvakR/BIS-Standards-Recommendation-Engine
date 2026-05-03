import os
from langchain_chroma import Chroma
import logging

from app.storage.manager import StorageFactory
from app.storage.main import Store
from app.embeddings.embedding import Embedding
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

@StorageFactory.register_store("chromadb")
class ChromaStore(Store):
    def __init__(
            self,
            namespace: str,
            persist_dir: str = os.getenv('PERSISTENT_STORAGE'),
        ):
        logger.info(f"Initializing ChromaStore with namespace: {namespace} and persist_dir: {persist_dir}")
        self.namespace = namespace
        self.persist_dir = persist_dir

        self.vector_store = Chroma(
            collection_name=namespace,
            persist_directory=persist_dir,
            embedding_function=Embedding().get_embedding_function(),
        )

    def get_vector_store(self) -> Chroma:
        return self.vector_store
        