<p align="center">
  <img src="logo.png" alt="Ragamuffin logo" width="280">
</p>

# Ragamuffin

Local semantic search for Markdown. Index your documents and retrieve ranked passages with source paths, headings, and metadata. Embeddings run locally through Ollama.

Built with Python, FastAPI, PostgreSQL/pgvector, and Ollama. Runs with Docker Compose.

## Get started

Requires Docker with Compose.

```bash
cp .env.example .env
```

Set `POSTGRES_PASSWORD` in `.env` and point `DOCUMENTS_PATH` at your Markdown directory. Keep the default to try the included examples.

```bash
chmod 600 .env
docker compose up -d --build
docker compose exec ollama ollama pull all-minilm
```

## Index and search

```bash
# Index documents; rerun after edits, additions, or deletions.
docker compose exec api .venv/bin/python -m app.ingestion.sync

# Search for evidence.
docker compose exec api .venv/bin/python -m app.retrieval.search \
  'What is the access control policy?' --top-k 5
```

Search returns passages and citations. Set `MIN_SIMILARITY` in `.env` to override the provisional cosine cutoff of 0.30. HTTP returns `no_evidence` when no passage qualifies. Review each citation for relevance, scope, and validity before using it in a questionnaire.

Sync skips unchanged files, replaces modified files atomically, and removes missing files after a successful run. Failed updates retain previous evidence and prevent deletion cleanup. Use one document root per index; syncing an empty directory clears its indexed content.

To preview extracted chunks, run sync with `--inspect`.

The HTTP equivalent uses the configured document mount:

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/index/sync
curl -fsS http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' -d '{"query":"What is the access control policy?"}'
```

## Run locally

```bash
docker compose up -d                  # Start
docker compose down                   # Stop
docker compose up -d --build api       # Rebuild after code changes
docker compose logs -f api             # View logs
curl -fsS http://localhost:8000/healthz # Check health
```

Documents are mounted read only. Database and model files persist in `data/`; keep these and `.env` out of Git. Indexing is explicit. The HTTP API is published on localhost.

## Tests

```bash
docker compose --profile test build test
docker compose --profile test run --rm --no-deps test
```

For the full suite, start the disposable test database and Ollama, wait for the database to be healthy, then run:

```bash
docker compose --profile test up -d postgres-test ollama
docker compose --profile test run --rm --no-deps test \
  .venv/bin/python -m pytest -v
```

Integration tests require the model downloaded during setup and use a separate test database.

Run the synthetic evaluation with the same test services:

```bash
docker compose --profile test run --rm --no-deps test \
  .venv/bin/python -m app.evaluation.run --dataset /app/tests/fixtures/evaluation.json
```

Add `--calibrate` to select a cutoff on the calibration split. If it passes, set `MIN_SIMILARITY` to the reported value and rerun without `--calibrate`. Use `--baseline` to compare runs with the same hardware, model, cutoff, and corpus.

For standalone Compose installations, use `docker-compose` in place of `docker compose`.

## Backups and upgrades

With Bash and Perl on the host, create and verify a backup from the repo directory:

```bash
mkdir -p backups
./scripts/db-backup.sh backup backups/index.dump
./scripts/db-backup.sh restore backups/index.dump rag_restore
```

Restore creates a new database. To use it, point the API's `DATABASE_URL` at that database and keep the same embedding model and document root. Dumps contain indexed text and embeddings. Store a protected copy outside the database host and back up the original Markdown separately.

Before upgrading, take and verify a dump, then rebuild with the new pins and check `/readyz` and a known search. Use a fresh database for reindexing after an incompatible model or chunking change. PostgreSQL major upgrades need a separate data directory and dump/restore.

Initial setup needs network access for images, build dependencies, and the model. With those cached, indexing and search run offline. The API trusts local clients and has no authentication. PostgreSQL and Ollama have no published ports. Add authentication before exposing the API remotely.

If `/readyz` fails, check `docker compose ps`, API logs, and `ollama list` inside the Ollama container. Use `docker stats` to check memory during indexing and search.

[MIT License](LICENSE)
