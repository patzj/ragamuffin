"""Search indexed Markdown passages using local query embeddings."""

import argparse
import json
import math
import sys

import psycopg
from pydantic import Field
from pydantic_settings import BaseSettings

from app.embeddings import create_embeddings
from app.ingestion.markdown import CHUNKING_VERSION
from app.ingestion.sync import IngestionSettings, report_error


class SearchSettings(BaseSettings, env_ignore_empty=True):
    # Provisional until the release calibration is run with real Ollama.
    min_similarity: float = Field(default=0.30, ge=-1, le=1)


def search(
    query: str, top_k: int = 5, *, database_url: str | None = None
) -> list[dict]:
    cutoff = SearchSettings().min_similarity
    return [
        hit
        for hit in rank_candidates(query, top_k, database_url=database_url)
        if hit["score"] >= cutoff
    ]


def rank_candidates(
    query: str, top_k: int = 5, *, database_url: str | None = None
) -> list[dict]:
    if not query.strip():
        raise ValueError("Query is required")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if database_url is None:
        database_url = IngestionSettings().database_url
    if not database_url:
        raise ValueError("DATABASE_URL is required")
    embeddings = create_embeddings()
    with psycopg.connect(database_url) as connection:
        # One snapshot keeps configuration and passages consistent across a sync.
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        if (
            connection.execute("SELECT to_regclass('ragamuffin_index')").fetchone()[0]
            is None
        ):
            raise ValueError("Index not initialized; run sync")
        row = connection.execute(
            "SELECT model, dimension, chunking_version FROM ragamuffin_index WHERE id = 1"
        ).fetchone()
        if row is None:
            raise ValueError("Index not initialized; run sync")
        model, dimension, chunking = row
        if (model, chunking) != (embeddings.model, CHUNKING_VERSION):
            raise ValueError("Incompatible index configuration")
        if not connection.execute(
            "SELECT EXISTS(SELECT 1 FROM ragamuffin_chunks)"
        ).fetchone()[0]:
            return []
        vector = embeddings.embed_query(query.strip())
        if (
            not vector
            or len(vector) != dimension
            or not all(math.isfinite(value) for value in vector)
            or not any(vector)
        ):
            raise ValueError("Invalid query embedding")
        rows = connection.execute(
            """SELECT id, document_id, content, metadata,
                      1 - (embedding <=> %s::vector) AS score
               FROM ragamuffin_chunks
               ORDER BY embedding <=> %s::vector, id LIMIT %s""",
            (str(vector), str(vector), top_k),
        ).fetchall()
        return [
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": text,
                "source": metadata["source"],
                "headings": metadata["headings"],
                "metadata": metadata,
                "score": score,
            }
            for rank, (chunk_id, document_id, text, metadata, score) in enumerate(
                rows, 1
            )
        ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        results = search(args.query, args.top_k)
        print(
            json.dumps(
                {"query": args.query.strip(), "results": results}, ensure_ascii=False
            )
        )
        return 0
    except Exception as error:  # noqa: BLE001 - concise CLI failure without credentials
        report_error("search", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
