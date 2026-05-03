from langchain_huggingface import HuggingFaceEmbeddings

class Embedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder="./cache"
        )

    def get_embedding_function(self):
        return self.embedding_function
    