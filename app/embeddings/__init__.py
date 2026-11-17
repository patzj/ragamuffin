from langchain_ollama import OllamaEmbeddings
from pydantic_settings import BaseSettings


class EmbeddingSettings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "all-minilm"


def create_embeddings() -> OllamaEmbeddings:
    settings = EmbeddingSettings()
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        # Release model memory after each on-demand embedding request.
        keep_alive=0,
        validate_model_on_init=False,
    )
