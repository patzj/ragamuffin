import json
import os
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from app.embeddings import create_embeddings
from app.ingestion import sync
from app.ingestion.markdown import extract
from app.ingestion.store import IngestionStore
from app.retrieval import search

pytestmark = pytest.mark.integration


@pytest.fixture
def index(monkeypatch):
    database_url = os.environ["TEST_DATABASE_URL"]
    schema = "test_" + uuid.uuid4().hex
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(f'CREATE SCHEMA "{schema}"')
    url = make_conninfo(database_url, options=f"-csearch_path={schema},public")
    monkeypatch.setenv("DATABASE_URL", url)
    store = IngestionStore(url, create_embeddings().model)
    try:
        yield store, url
    finally:
        store.close()


def test_search_and_incremental_lifecycle(index, tmp_path, capsys):
    store, _url = index
    assert search.search("access") == []
    fixtures = (
        Path("/fixtures") if Path("/fixtures").exists() else Path("examples/documents")
    )
    for source in fixtures.glob("*.md"):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    assert sync.main(["--path", str(tmp_path)]) == 0
    capsys.readouterr()
    results = search.search("What is the access control policy?", 2)
    assert len(results) == 2
    assert results[0]["source"] == "access.md"
    assert results[0]["headings"][0] == "Access control"
    assert [r["rank"] for r in results] == [1, 2]
    assert results[0]["score"] >= results[1]["score"]
    assert search.main(["What is the access control policy?", "--top-k", "1"]) == 0
    assert len(json.loads(capsys.readouterr().out)["results"]) == 1
    cert = next(
        r
        for r in search.search("certificate", 100)
        if r["source"] == "certification.md"
    )
    assert cert["metadata"]["frontmatter"]["expires_on"] == "2026-01-01"
    assert sync.main(["--path", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().err.splitlines()[-1])["skipped"] == 3

    # A failed new/renamed file must keep all missing-file evidence for recovery.
    (tmp_path / "access.md").rename(tmp_path / "renamed.md")
    (tmp_path / "renamed.md").write_text("---\ninvalid")
    assert sync.main(["--path", str(tmp_path)]) == 1
    assert (
        store.connection.execute(
            "SELECT count(*) FROM ragamuffin_documents WHERE source='access.md'"
        ).fetchone()[0]
        == 1
    )
    (tmp_path / "renamed.md").write_text("# Replacement\nNew synthetic evidence")
    assert sync.main(["--path", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().err.splitlines()[-1])["deleted"] == 1
    sources = {r["source"] for r in search.rank_candidates("evidence", 100)}
    assert "access.md" not in sources and "renamed.md" in sources

    # Modified-to-empty, deletion, and idempotent reruns.
    (tmp_path / "renamed.md").write_text("")
    (tmp_path / "architecture.md").unlink()
    assert sync.main(["--path", str(tmp_path)]) == 0
    assert {r["source"] for r in search.rank_candidates("evidence", 100)} == {
        "certification.md"
    }
    for path in tmp_path.glob("*.md"):
        path.unlink()
    assert sync.main(["--path", str(tmp_path)]) == 0
    assert search.search("evidence") == []
    assert sync.main(["--path", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().err.splitlines()[-1])["deleted"] == 0


def test_transactions_and_lock(index, tmp_path):
    store, url = index
    path = tmp_path / "one.md"
    path.write_text("# Old\nOriginal")
    old = extract(path, tmp_path)
    store.add(old, [[1.0, 0.0]])
    other_path = tmp_path / "two.md"
    other_path.write_text("# Other\nOther")
    other = extract(other_path, tmp_path)
    store.add(other, [[0.0, 1.0]])
    path.write_text("# New\nReplacement")
    new = extract(path, tmp_path)
    new.chunks[0]["id"] = other.chunks[0]["id"]
    with pytest.raises(psycopg.errors.UniqueViolation):
        store.add(new, [[1.0, 0.0]])
    assert store.unchanged(old)
    assert (
        store.connection.execute("SELECT count(*) FROM ragamuffin_chunks").fetchone()[0]
        == 2
    )
    new = extract(path, tmp_path)
    with psycopg.connect(url) as reader:
        reader.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        assert (
            reader.execute(
                "SELECT content FROM ragamuffin_chunks WHERE document_id=%s",
                (old.document_id,),
            ).fetchone()[0]
            == old.chunks[0]["text"]
        )
        assert store.add(new, [[1.0, 0.0]])
        assert (
            reader.execute(
                "SELECT content FROM ragamuffin_chunks WHERE document_id=%s",
                (old.document_id,),
            ).fetchone()[0]
            == old.chunks[0]["text"]
        )
    assert not store.add(new, [[1.0, 0.0]])

    # Force deletion to fail after child rows are removed; the transaction restores them.
    store.connection.execute(
        "CREATE TABLE deletion_guard (id text REFERENCES ragamuffin_documents(id))"
    )
    store.connection.execute(
        "INSERT INTO deletion_guard VALUES (%s)", (old.document_id,)
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        store.remove_missing([])
    assert (
        store.connection.execute("SELECT count(*) FROM ragamuffin_chunks").fetchone()[0]
        == 2
    )

    store.acquire_sync_lock()
    competitor = IngestionStore(url, store.model)
    try:
        with pytest.raises(ValueError, match="already running"):
            competitor.acquire_sync_lock()
    finally:
        competitor.close()


def test_ranking_validation_and_failed_sync(index, tmp_path, monkeypatch):
    store, _url = index
    path = tmp_path / "one.md"
    path.write_text("# Original\nEvidence")
    old = extract(path, tmp_path)
    store.add(old, [[1.0, 0.0]])
    second = tmp_path / "two.md"
    second.write_text("# Second\nEvidence")
    other = extract(second, tmp_path)
    store.add(other, [[1.0, 0.0]])

    class Embeddings:
        model = store.model

        def __init__(self):
            self.vector = [1.0, 0.0]

        def embed_query(self, query):
            return self.vector

        def embed_documents(self, texts):
            raise RuntimeError("Embedding failed")

    embeddings = Embeddings()
    monkeypatch.setattr(search, "create_embeddings", lambda: embeddings)
    ranked = search.search("evidence", 100)
    assert [row["chunk_id"] for row in ranked] == sorted(
        [old.chunks[0]["id"], other.chunks[0]["id"]]
    )
    assert all(row["score"] == 1 for row in ranked)
    for invalid in ([0.0, 0.0], [float("nan"), 1.0], [1.0], []):
        embeddings.vector = invalid
        with pytest.raises(ValueError, match="Invalid query"):
            search.search("evidence")

    # Embedding failures retain old evidence and suppress missing-file pruning.
    path.write_text("# Changed\nNew evidence")
    second.unlink()
    monkeypatch.setattr(sync, "create_embeddings", lambda: embeddings)
    assert sync.main(["--path", str(tmp_path)]) == 1
    assert store.unchanged(old)
    assert store.unchanged(other)

    def failed_scan(root):
        raise PermissionError("Incomplete scan")

    monkeypatch.setattr(sync, "discover", failed_scan)
    assert sync.main(["--path", str(tmp_path)]) == 1
    assert store.unchanged(other)
    embeddings.model = "wrong-model"
    with pytest.raises(ValueError, match="Incompatible"):
        search.search("evidence")


def test_http_lifecycle(index, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    store, _url = index
    client = TestClient(app)
    monkeypatch.setenv("DOCUMENTS_DIR", str(tmp_path))
    with store.connection.transaction():
        store.connection.execute("DROP TABLE ragamuffin_chunks")
        store.connection.execute("DROP TABLE ragamuffin_documents")
        store.connection.execute("DROP TABLE ragamuffin_index")
    assert client.get("/readyz").json() == {"status": "ready"}
    assert client.get("/v1/stats").json()["initialized"] is False
    query = {"query": "Who may access the service?"}
    assert client.post("/v1/search", json=query).status_code == 503
    assert client.post("/v1/index/sync").status_code == 200
    assert client.post("/v1/search", json=query).json()["status"] == "no_evidence"

    path = tmp_path / "access.md"
    path.write_text(
        "# Access policy\nOnly approved employees may access the service.\n"
    )
    assert client.post("/v1/index/sync").json()["indexed"] == 1
    stats = client.get("/v1/stats").json()
    assert stats["documents"] == stats["chunks"] == 1
    evidence = client.post("/v1/search", json=query).json()["results"][0]
    assert evidence["source"] == "access.md"
    assert evidence["headings"] == ["Access policy"]

    store.acquire_sync_lock()
    assert client.post("/v1/index/sync").status_code == 409
    store.connection.execute("SELECT pg_advisory_unlock_all()")
    path.rename(tmp_path / "invalid.md")
    (tmp_path / "invalid.md").write_text("---\n")
    response = client.post("/v1/index/sync")
    assert response.status_code == 500
    assert response.json()["failed"] == 1
    assert client.get("/v1/stats").json()["documents"] == 1

    monkeypatch.setenv("EMBEDDING_MODEL", "missing-model")
    assert client.get("/readyz").json() == {"detail": "Embedding model unavailable"}
    assert client.post("/v1/search", json=query).status_code == 503
