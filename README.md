<p align="center">
  <img src="logo.png" alt="Ragamuffin logo" width="280">
</p>

# Ragamuffin

Local Markdown search with ranked passages and citations. Ragamuffin uses
Python, FastAPI, PostgreSQL with pgvector, and Ollama.

## Quick start

From the repo directory, copy the environment file:

```bash
cp .env.example .env
chmod 600 .env
```

Set `POSTGRES_PASSWORD` in `.env`. Keep the default `DOCUMENTS_PATH` to try the
bundled documents, or point it at your Markdown directory.

Start Docker Compose, download the model, and search:

```bash
docker compose up -d --build
docker compose exec ollama ollama pull all-minilm
docker compose exec api .venv/bin/python -m app.ingestion.sync
docker compose exec api .venv/bin/python -m app.retrieval.search \
  'What is the access control policy?' --top-k 5
```

For standalone Compose installations, substitute `docker-compose` for
`docker compose`.

## Usage

### Sync documents

Run sync after adding, editing, or deleting files. Sync
skips unchanged files, replaces modified files atomically, and removes missing
files after a successful run. Failed updates retain previous evidence and
prevent deletion cleanup. Use one document root per index. Syncing an empty
directory clears its indexed content.

Use `--inspect` to preview chunks. To sync the document mount over HTTP:

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/index/sync
```

### Search the index

```bash
curl -fsS http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the access control policy?","top_k":5}'
```

Set `MIN_SIMILARITY` in `.env` and run `docker compose up -d` to override the
provisional cosine similarity cutoff of 0.30. The HTTP API returns
`no_evidence` when no passage qualifies. Review citations for relevance, scope,
and validity before using them in a questionnaire.

## Operations

```bash
docker compose up -d                    # Start
docker compose down                     # Stop
docker compose up -d --build api         # Rebuild after code changes
docker compose logs -f api               # View API logs
curl -fsS http://127.0.0.1:8000/healthz   # Check API liveness
curl -fsS http://127.0.0.1:8000/readyz    # Check database and model availability
```

If readiness fails, check `docker compose ps`, the API logs, and
`docker compose exec ollama ollama list`. Use `docker stats` to check memory
during indexing and search.

The API binds to localhost and trusts local clients. Add authentication before
exposing it remotely. PostgreSQL and Ollama have no published ports. The
document mount is read-only. Database and model files persist in `data/`.
Keep `data/` and `.env` out of Git. Setup needs network access for images, build
dependencies, and the model. Indexing and search run offline once those are
cached.

### Backups and upgrades

With Bash and Perl on the host, create a backup and verify it by restoring into
a new database:

```bash
mkdir -p backups
./scripts/db-backup.sh backup backups/index.dump
./scripts/db-backup.sh restore backups/index.dump rag_restore
```

To use the restored index, update the API's `DATABASE_URL` and keep the same
embedding model and document root. Dumps contain indexed text and embeddings.
Keep a protected copy off-host and back up the original Markdown separately.

Before upgrading, take and verify a dump. Rebuild with the new pins, then check
`/readyz` and run a known search. Use a fresh database to reindex after an
incompatible model or chunking change. PostgreSQL major upgrades need a
separate data directory and a dump and restore.

## Development

Build the test image and run the fast suite:

```bash
docker compose --profile test build test
docker compose --profile test run --rm --no-deps test
```

The full suite uses the model from Quick start and a disposable database. Wait
for `postgres-test` to report healthy in `docker compose --profile test ps`
before running pytest.

```bash
docker compose --profile test up -d postgres-test ollama
docker compose --profile test run --rm --no-deps test \
  .venv/bin/python -m pytest -v
```

Run the synthetic retrieval evaluation with the same services:

```bash
docker compose --profile test run --rm --no-deps test \
  .venv/bin/python -m app.evaluation.run --dataset /app/tests/fixtures/evaluation.json
```

Add `--calibrate` to select a cutoff on the calibration split. If calibration
passes, set `MIN_SIMILARITY` to the reported value and rerun on the held-out split
by omitting `--calibrate`. Use `--baseline` to compare runs with the same
hardware, model, cutoff, and corpus.

[MIT License](LICENSE)
