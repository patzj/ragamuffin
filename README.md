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

Search returns JSON passages and citations. It generates no answers and applies no relevance or expiry cutoff.

Sync skips unchanged files, replaces modified files atomically, and removes missing files after a successful run. Failed updates retain previous evidence and prevent deletion cleanup. Use one document root per index; syncing an empty directory clears its indexed content.

To preview extracted chunks, run sync with `--inspect`.

## Run locally

```bash
docker compose up -d                  # Start
docker compose down                   # Stop
docker compose up -d --build api       # Rebuild after code changes
docker compose logs -f api             # View logs
curl -fsS http://localhost:8000/healthz # Check health
```

Documents are mounted read only. Database and model files persist in `data/`; keep these and `.env` out of Git. Indexing is explicit, and retrieval currently uses the CLI.

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

For standalone Compose installations, use `docker-compose` in place of `docker compose`.
