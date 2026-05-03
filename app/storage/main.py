from abc import ABC, abstractmethod

class Store(ABC):

    @abstractmethod
    def get_vector_store(self):
        pass