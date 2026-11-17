from base64 import b64encode
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from ollama import Client
from pydantic_settings import BaseSettings


class EmbeddingSettings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "all-minilm"


@dataclass
class OllamaEmbeddings:
    base_url: str
    model: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        url = urlsplit(self.base_url)
        headers = {}
        if url.username is not None:
            credentials = f"{unquote(url.username)}:{unquote(url.password or '')}"
            headers["Authorization"] = (
                "Basic " + b64encode(credentials.encode()).decode()
            )
        with Client(host=self.base_url, headers=headers) as client:
            # Release model memory after each on-demand embedding request.
            return client.embed(self.model, texts, keep_alive=0).embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embeddings() -> OllamaEmbeddings:
    settings = EmbeddingSettings()
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
    )
