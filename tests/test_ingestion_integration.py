import json
import os
import uuid
from pathlib import Path

import httpx
import psycopg
import pytest
from psycopg.conninfo import make_conninfo
from test_embeddings import assert_model_unloaded

from app.embeddings import EmbeddingSettings, create_embeddings
from app.ingestion import sync
from app.ingestion.markdown import extract
from app.ingestion.store import IngestionStore

pytestmark = pytest.mark.integration


def test_pgvector_ingestion(tmp_path, monkeypatch, capsys):
    database_url = os.environ["TEST_DATABASE_URL"]
    schema = "test_" + uuid.uuid4().hex
    # Only the explicitly configured disposable test database is touched.
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(f'CREATE SCHEMA "{schema}"')
    url = make_conninfo(database_url, options=f"-csearch_path={schema},public")
    embeddings = create_embeddings()
    store = IngestionStore(url, embeddings.model)
    try:
        root = (
            Path("/fixtures")
            if Path("/fixtures").exists()
            else Path("examples/documents")
        )
        source = root / "certification.md"
        local = tmp_path / source.name
        local.write_bytes(source.read_bytes())
        document = extract(local, tmp_path)
        vectors = embeddings.embed_documents(
            [chunk["text"] for chunk in document.chunks]
        )
        assert store.add(document, vectors)
        assert store.unchanged(document)
        assert not store.add(document, vectors)
        rows = store.connection.execute(
            "SELECT content, metadata, vector_dims(embedding) FROM ragamuffin_chunks ORDER BY metadata->>'chunk_index'"
        ).fetchall()
        assert len(rows) == len(document.chunks)
        for (text, metadata, dimension), chunk in zip(
            rows, document.chunks, strict=True
        ):
            assert text == chunk["text"]
            assert metadata == chunk["metadata"]
            assert dimension == len(vectors[0])
            assert metadata["frontmatter"]["source"].endswith("example-2025.pdf")
            assert metadata["frontmatter"]["expires_on"] == "2026-01-01"
        local.write_text("# Changed\nNew text")
        assert not store.unchanged(extract(local, tmp_path))
        bad = tmp_path / "bad.md"
        bad.write_text("# New\nText")
        rejected = extract(bad, tmp_path)
        with pytest.raises(ValueError, match="dimension"):
            store.add(rejected, [[1.0]])
        assert not store.unchanged(rejected)
        with pytest.raises(ValueError, match="Invalid embedding"):
            store.add(rejected, [[float("nan")]])
        # Force an actual insert failure after the document row has been written.
        rejected.chunks[0]["id"] = document.chunks[0]["id"]
        with pytest.raises(psycopg.errors.UniqueViolation):
            store.add(rejected, [vectors[0]])
        assert not store.unchanged(rejected)
        assert (
            store.connection.execute(
                "SELECT count(*) FROM ragamuffin_documents"
            ).fetchone()[0]
            == 1
        )
        with pytest.raises(ValueError, match="Incompatible"):
            IngestionStore(url, "different-model")
        # Exercise the public CLI against real services, including identical reruns.
        local.write_bytes(source.read_bytes())
        bad.write_text("# New\nAdditional synthetic evidence")
        monkeypatch.setenv("DATABASE_URL", url)
        for indexed, skipped in ((1, 1), (0, 2)):
            assert sync.main(["--path", str(tmp_path)]) == 0
            counts = json.loads(capsys.readouterr().err.splitlines()[-1])
            assert counts["indexed"] == indexed
            assert counts["skipped"] == skipped
            assert counts["failed"] == 0
        local.write_text("# Changed\nNew evidence")
        assert sync.main(["--path", str(tmp_path)]) == 0
        assert json.loads(capsys.readouterr().err.splitlines()[-1])["indexed"] == 1
        settings = EmbeddingSettings()
        with httpx.Client(base_url=settings.ollama_base_url) as client:
            assert_model_unloaded(client, settings.embedding_model)
    finally:
        store.close()
