import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


@pytest.mark.parametrize(
    "body",
    [
        {"query": " "},
        {"query": "q", "top_k": True},
        {"query": "q", "top_k": "5"},
        {"query": "q", "top_k": 0},
        {"query": "q", "top_k": 101},
    ],
)
def test_invalid_search(body):
    assert TestClient(app).post("/v1/search", json=body).status_code == 422


def test_search_empty_and_backend_failure(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(routes, "search", lambda *args: [])
    assert client.post("/v1/search", json={"query": " test "}).json() == {
        "query": "test",
        "status": "no_evidence",
        "results": [],
    }

    def unavailable(*args):
        raise ConnectionError("private backend details")

    monkeypatch.setattr(routes, "search", unavailable)
    response = client.post("/v1/search", json={"query": "test"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Retrieval unavailable"}


def test_readiness_failure(monkeypatch):
    client = TestClient(app)

    def unavailable():
        raise OSError("private backend details")

    monkeypatch.setenv("DATABASE_URL", "")
    assert client.get("/readyz").json() == {"detail": "PostgreSQL unavailable"}
    monkeypatch.setattr(routes, "connect", unavailable)
    assert client.get("/readyz").status_code == 503
    assert client.get("/healthz").status_code == 200


def test_model_readiness_failure_is_logged_safely(monkeypatch, caplog):
    from contextlib import nullcontext
    from unittest.mock import Mock

    monkeypatch.setattr(routes, "connect", lambda: nullcontext(Mock()))

    def unavailable():
        raise OSError("private model details")

    monkeypatch.setattr(routes, "check_model", unavailable)
    response = TestClient(app).get("/readyz?query=private-query")
    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding model unavailable"}
    assert '"failure": "model_unavailable"' in caplog.text
    assert '"route": "/readyz"' in caplog.text
    assert '"status": 503' in caplog.text
    assert "private" not in caplog.text


def test_unexpected_sync_failure_is_sanitized(monkeypatch, caplog):
    def unavailable(*args, **kwargs):
        raise RuntimeError("private document /credentials.md")

    monkeypatch.setattr(routes, "run_sync", unavailable)
    response = TestClient(app).post("/v1/index/sync")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert '"failure": "unexpected_error"' in caplog.text
    assert "private" not in caplog.text
    assert "credentials.md" not in caplog.text
