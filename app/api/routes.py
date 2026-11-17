"""Local HTTP endpoints for Markdown retrieval and indexing."""

import json
from pathlib import Path
from urllib.request import Request as URLRequest
from urllib.request import urlopen

import psycopg
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.embeddings import EmbeddingSettings
from app.ingestion.store import SyncInProgress
from app.ingestion.sync import IngestionSettings, run_sync
from app.retrieval.search import search

router = APIRouter()


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, strict=True, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def reject_nul(cls, value: str) -> str:
        if "\0" in value:
            raise ValueError("Query contains a null character")
        return value


@router.post("/v1/search")
def search_index(request: Request, body: SearchRequest):
    try:
        results = search(body.query, body.top_k)
    except Exception as error:  # noqa: BLE001 - keep backend details private
        request.state.failure = "retrieval_unavailable"
        message = "Retrieval unavailable"
        if type(error) is ValueError and str(error) in {
            "Index not initialized; run sync",
            "Incompatible index configuration",
        }:
            message = str(error)
        raise HTTPException(503, message) from None
    return {
        "query": body.query,
        "status": "ok" if results else "no_evidence",
        "results": results,
    }


@router.post("/v1/index/sync")
def sync_index(request: Request):
    settings = IngestionSettings()
    root = Path(settings.documents_dir)
    counts, errors = run_sync(root, database_url=settings.database_url)
    if any(isinstance(error, SyncInProgress) for _, error in errors):
        raise HTTPException(409, "Sync already running")
    if errors:
        request.state.failure = "sync_failed"
    return JSONResponse(
        {
            **counts,
            "errors": [
                {
                    "source": None if source == str(root) else source,
                    "error": type(error).__name__,
                }
                for source, error in errors
            ],
        },
        status_code=500 if errors else 200,
    )


def connect():
    url = IngestionSettings().database_url
    if not url:
        raise ValueError("DATABASE_URL is required")
    return psycopg.connect(url, connect_timeout=2)


@router.get("/v1/stats")
def index_stats(request: Request):
    stats = {
        "initialized": False,
        "documents": 0,
        "chunks": 0,
        "model": None,
        "dimension": None,
    }
    try:
        with connect() as connection:
            connection.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
            )
            if connection.execute("SELECT to_regclass('ragamuffin_index')").fetchone()[
                0
            ]:
                row = connection.execute("""
                    SELECT model, dimension,
                        (SELECT count(*) FROM ragamuffin_documents),
                        (SELECT count(*) FROM ragamuffin_chunks)
                    FROM ragamuffin_index WHERE id = 1
                """).fetchone()
                if row is not None:
                    model, dimension, documents, chunks = row
                    stats.update(
                        initialized=True,
                        model=model,
                        dimension=dimension,
                        documents=documents,
                        chunks=chunks,
                    )
    except Exception:  # noqa: BLE001 - keep backend details private
        request.state.failure = "stats_unavailable"
        raise HTTPException(503, "Index stats unavailable") from None
    return stats


def check_model():
    settings = EmbeddingSettings()
    request = URLRequest(
        settings.ollama_base_url.rstrip("/") + "/api/show",
        data=json.dumps({"model": settings.embedding_model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=2):
        pass


@router.get("/readyz")
def readiness(request: Request):
    try:
        with connect() as connection:
            connection.execute("SET LOCAL statement_timeout = '2s'")
            connection.execute("SELECT 1")
    except Exception:  # noqa: BLE001 - keep backend details private
        request.state.failure = "postgres_unavailable"
        raise HTTPException(503, "PostgreSQL unavailable") from None
    try:
        check_model()
    except Exception:  # noqa: BLE001 - keep backend details private
        request.state.failure = "model_unavailable"
        raise HTTPException(503, "Embedding model unavailable") from None
    return {"status": "ready"}
