"""Explicit local Markdown indexing and extraction inspection."""

import argparse
import json
import sys
from pathlib import Path

from pydantic_settings import BaseSettings

from app.embeddings import create_embeddings
from app.ingestion.markdown import discover, extract
from app.ingestion.store import IngestionStore


class IngestionSettings(BaseSettings):
    documents_dir: str = "/documents"
    database_url: str = ""


def report_error(source: str, error: Exception):
    # Database/HTTP exception strings can contain credentials or document bodies.
    message = (
        str(error)
        if isinstance(error, (ValueError, UnicodeError))
        else type(error).__name__
    )
    print(json.dumps({"source": source, "error": message}), file=sys.stderr)


def run_sync(path, *, database_url, inspect=False, on_chunk=None):
    errors = []
    counts = {"indexed": 0, "skipped": 0, "failed": 0, "inspected": 0, "deleted": 0}
    store = None
    try:
        embeddings = None
        if not inspect:
            if not database_url:
                raise ValueError("DATABASE_URL is required")
            embeddings = create_embeddings()
            store = IngestionStore(database_url, embeddings.model, lock_for_sync=True)
        paths = discover(path)
        for file in paths:
            try:
                document = extract(file, path)
                if inspect:
                    for chunk in document.chunks:
                        on_chunk(chunk)
                    counts["inspected"] += 1
                elif store.unchanged(document):
                    counts["skipped"] += 1
                else:
                    vectors = []
                    # Small requests keep embedding memory bounded under the Ollama cap.
                    for start in range(0, len(document.chunks), 16):
                        vectors.extend(
                            embeddings.embed_documents(
                                [
                                    chunk["text"]
                                    for chunk in document.chunks[start : start + 16]
                                ]
                            )
                        )
                    counts[
                        "indexed" if store.add(document, vectors) else "skipped"
                    ] += 1
            except Exception as error:  # noqa: BLE001 - report a file failure and continue
                counts["failed"] += 1
                errors.append((file.relative_to(path).as_posix(), error))
        if store and not counts["failed"]:
            counts["deleted"] = store.remove_missing(
                [file.relative_to(path).as_posix() for file in paths]
            )
    except Exception as error:  # noqa: BLE001 - report a file failure and continue
        counts["failed"] += 1
        errors.append((str(path), error))
    finally:
        if store:
            store.close()
    return counts, errors


def main(argv=None) -> int:
    settings = IngestionSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path(settings.documents_dir))
    parser.add_argument(
        "--inspect", action="store_true", help="Print chunks without network requests"
    )
    args = parser.parse_args(argv)
    counts, errors = run_sync(
        args.path,
        database_url=settings.database_url,
        inspect=args.inspect,
        on_chunk=lambda chunk: print(json.dumps(chunk, ensure_ascii=False)),
    )
    for source, error in errors:
        report_error(source, error)
    print(json.dumps(counts), file=sys.stderr)
    return int(counts["failed"] > 0)


if __name__ == "__main__":
    sys.exit(main())
