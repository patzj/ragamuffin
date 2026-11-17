"""Small pgvector store; each file is committed as a unit."""

import math

import psycopg
from psycopg.types.json import Jsonb

from app.ingestion.markdown import CHUNKING_VERSION, ExtractedDocument


class IngestionStore:
    def __init__(self, database_url: str, model: str):
        self.connection = psycopg.connect(database_url, autocommit=True)
        self.model = model
        try:
            with self.connection.transaction():
                self.connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS ragamuffin_index (
                        id integer PRIMARY KEY CHECK (id = 1),
                        model text NOT NULL, dimension integer,
                        chunking_version text NOT NULL
                    )
                """)
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS ragamuffin_documents (
                        id text PRIMARY KEY, source text NOT NULL UNIQUE,
                        source_digest text NOT NULL
                    )
                """)
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS ragamuffin_chunks (
                        id text PRIMARY KEY,
                        document_id text NOT NULL REFERENCES ragamuffin_documents(id),
                        content text NOT NULL, embedding vector NOT NULL,
                        metadata jsonb NOT NULL
                    )
                """)
                self.connection.execute(
                    "INSERT INTO ragamuffin_index VALUES (1, %s, NULL, %s) ON CONFLICT DO NOTHING",
                    (model, CHUNKING_VERSION),
                )
                self._configuration()
        except Exception:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def _configuration(self):
        model, dimension, chunking = self.connection.execute(
            "SELECT model, dimension, chunking_version FROM ragamuffin_index WHERE id = 1 FOR UPDATE"
        ).fetchone()
        if (model, chunking) != (self.model, CHUNKING_VERSION):
            raise ValueError("Incompatible index configuration")
        return dimension

    def unchanged(self, document: ExtractedDocument) -> bool:
        row = self.connection.execute(
            "SELECT source_digest FROM ragamuffin_documents WHERE id = %s",
            (document.document_id,),
        ).fetchone()
        if row and row[0] != document.source_digest:
            raise ValueError("Changed indexed file; replacement requires Phase 5")
        return row is not None

    def add(self, document: ExtractedDocument, vectors: list[list[float]]) -> bool:
        if len(vectors) != len(document.chunks):
            raise ValueError("Embedding count mismatch")
        dimension = len(vectors[0]) if vectors else None
        if vectors and (
            not dimension
            or any(
                len(vector) != dimension
                or not all(math.isfinite(value) for value in vector)
                for vector in vectors
            )
        ):
            raise ValueError("Invalid embedding vectors")
        with self.connection.transaction():
            # The index row serializes configuration checks and concurrent file inserts.
            expected = self._configuration()
            if self.unchanged(document):
                return False
            if dimension is not None:
                if expected is not None and expected != dimension:
                    raise ValueError("Embedding dimension mismatch")
                self.connection.execute(
                    "UPDATE ragamuffin_index SET dimension = %s WHERE id = 1",
                    (dimension,),
                )
            self.connection.execute(
                "INSERT INTO ragamuffin_documents VALUES (%s, %s, %s)",
                (document.document_id, document.source, document.source_digest),
            )
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO ragamuffin_chunks VALUES (%s, %s, %s, %s::vector, %s)",
                    [
                        (
                            chunk["id"],
                            document.document_id,
                            chunk["text"],
                            str(vector),
                            Jsonb(chunk["metadata"]),
                        )
                        for chunk, vector in zip(document.chunks, vectors, strict=True)
                    ],
                )
        return True
