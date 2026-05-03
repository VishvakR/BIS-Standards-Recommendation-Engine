from langchain_ollama import OllamaLLM
from app.llm.main import LanguageModel
from app.llm.manager import LanguageModelFactory

@LanguageModelFactory.register("opensource")
class OpenSourceModel(LanguageModel):
    def __init__(
            self, 
            model_name: str = "llama3.1:8b", 
            temperature: float = 0.7
        ):
        self.model_name = model_name
        self.temperature = temperature

        self.llm = OllamaLLM(
            model=model_name, 
            temperature=temperature
        )

    def generate_response(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response  # OllamaLLM returns a string directly

