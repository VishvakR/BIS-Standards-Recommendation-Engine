from abc import ABC, abstractmethod

class LanguageModel(ABC):
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass
