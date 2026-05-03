from typing import Optional
from langchain_openai import ChatOpenAI
from app.llm.main import LanguageModel
from app.llm.manager import LanguageModelFactory


@LanguageModelFactory.register("openai")
class OpenAIModel(LanguageModel):
    """LangChain-backed OpenAI chat model"""

    def __init__(
            self,
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.7,
            api_key: Optional[str] = None,
        ):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    def generate_response(self, prompt: str) -> str:
        """Invoke the chat model and return the reply text."""
        response = self.llm.invoke(prompt)
        return response.content

