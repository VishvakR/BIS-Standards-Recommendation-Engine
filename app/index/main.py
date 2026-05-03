from abc import ABC, abstractmethod


class IndexBase(ABC):
    @abstractmethod
    def create_index(self):
        """Create and persist the index from the document handler."""
        pass

    @abstractmethod
    def query(self, question: str):
        """Retrieve documents relevant to *question*."""
        pass