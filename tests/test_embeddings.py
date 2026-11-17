import importlib
import json
import math
import socket
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from ollama import Client

from app.embeddings import EmbeddingSettings, create_embeddings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, request):
    if request.node.get_closest_marker("integration"):
        return

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    def forbid_network(*args, **kwargs):
        pytest.fail("Unexpected network request")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)


def test_create_embeddings_defaults():
    embeddings = create_embeddings()

    assert embeddings.base_url == "http://localhost:11434"
    assert embeddings.model == "all-minilm"


def test_create_embeddings_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("EMBEDDING_MODEL", "sample-model")

    embeddings = create_embeddings()

    assert embeddings.base_url == "http://ollama:11434"
    assert embeddings.model == "sample-model"


@pytest.mark.parametrize(
    ("base_url", "authorization"),
    [
        ("http://localhost:11434", None),
        ("http://rag:local%20secret@localhost:11434", "Basic cmFnOmxvY2FsIHNlY3JldA=="),
    ],
)
def test_embedding_requests(monkeypatch, base_url, authorization):
    inputs = []
    monkeypatch.setenv("OLLAMA_BASE_URL", base_url)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    def respond(request):
        assert str(request.url) == "http://localhost:11434/api/embed"
        assert request.headers.get("Authorization") == authorization
        payload = json.loads(request.content)
        assert payload["model"] == "all-minilm"
        assert payload["keep_alive"] == 0
        inputs.append(payload["input"])
        return httpx.Response(
            200, json={"model": "all-minilm", "embeddings": [[1.0]] * len(inputs[-1])}
        )

    monkeypatch.setattr(
        "app.embeddings.Client",
        lambda **kwargs: Client(transport=httpx.MockTransport(respond), **kwargs),
    )
    embeddings = create_embeddings()
    assert embeddings.embed_documents(["one", "two"]) == [[1.0], [1.0]]
    assert embeddings.embed_query("query") == [1.0]
    assert inputs == [["one", "two"], ["query"]]


def test_health_without_ollama():
    from app import main

    importlib.reload(main)
    with TestClient(main.app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def assert_model_unloaded(client: httpx.Client, model: str) -> None:
    # Ollama reports implicit tags explicitly, e.g. all-minilm:latest.
    model_name = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
    deadline = time.monotonic() + 10
    loaded_models = []
    while time.monotonic() < deadline:
        response = client.get(
            "/api/ps", timeout=min(2, max(0.01, deadline - time.monotonic()))
        )
        response.raise_for_status()
        loaded_models = response.json()["models"]
        if not any(
            item.get(key) in {model, model_name}
            for item in loaded_models
            for key in ("name", "model")
        ):
            return
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    pytest.fail(f"Model did not unload: {loaded_models}")


@pytest.mark.integration
def test_real_embeddings_and_model_unload():
    settings = EmbeddingSettings()
    embeddings = create_embeddings()

    with httpx.Client(base_url=settings.ollama_base_url, timeout=2) as client:
        vectors = embeddings.embed_documents(
            ["Access requires an approved account.", "Backups run every night."]
        )
        assert len(vectors) == 2
        dimension = len(vectors[0])
        assert dimension > 0
        assert all(len(vector) == dimension for vector in vectors)
        assert all(math.isfinite(value) for vector in vectors for value in vector)
        assert_model_unloaded(client, settings.embedding_model)

        query = embeddings.embed_query("Who can access the service?")
        assert len(query) == dimension
        assert all(math.isfinite(value) for value in query)
        assert_model_unloaded(client, settings.embedding_model)
