<p align="center">
  <img src="logo.png" alt="Ragamuffin logo" width="280">
</p>

# Ragamuffin

A lightweight, fully local RAG service for Markdown documents, built with FastAPI, PostgreSQL/pgvector, and Ollama.

Markdown ingestion, local embeddings, and the health endpoint are ready. Semantic retrieval is still in development.

## Development

Development runs locally using Docker Compose. Install Docker with Compose before starting.

### First setup

```bash
cp .env.example .env
```

Edit `.env` and replace `POSTGRES_PASSWORD=replace-me` with a strong password. Set `DOCUMENTS_PATH` to your Markdown directory, or keep the default example directory.

```bash
chmod 600 .env
docker compose up -d --build
docker compose exec postgres psql -U rag -d rag -c 'CREATE EXTENSION IF NOT EXISTS vector;'
docker compose exec ollama ollama pull all-minilm
```

Open <http://localhost:8000/docs>, or check the API:

```bash
curl -fsS http://localhost:8000/healthz
```

### Make changes

Edit the Python files, then rebuild the API:

```bash
docker compose up -d --build api
docker compose logs -f api
```

Source code is copied into the image, so changes require a rebuild.

### Start and stop

```bash
docker compose up -d
docker compose down
```

Database and model files persist in `data/`. Documents are mounted read only. Keep `.env`, private documents, and runtime data out of Git.

If your installation uses standalone Compose, replace `docker compose` with `docker-compose`.

## Local embeddings and tests

Use `app.embeddings.create_embeddings()` to construct a LangChain `OllamaEmbeddings`
client. It reads `OLLAMA_BASE_URL` (default `http://localhost:11434`) and
`EMBEDDING_MODEL` (default `all-minilm`). Compose configures the client to use
`http://ollama:11434` within the Docker network. Construction makes no requests;
call `embed_documents(texts)` or `embed_query(text)` explicitly. Each request sets
`keep_alive=0` so Ollama releases model memory afterward.

Tests run in a separate Docker image with the locked development dependencies.
The production image excludes tests and development dependencies. Build the test
image and run the isolated tests (no embedding requests):

```bash
docker compose --profile test build test
docker compose --profile test run --rm --no-deps test
```

For integration verification, start Ollama and confirm that the model from first
setup is available. The tests do not download models. Wait for `ollama list` to
succeed before running them:

```bash
docker compose up -d ollama
docker compose exec ollama ollama list
docker compose --profile test run --rm test \
  .venv/bin/python -m pytest tests/test_embeddings.py -m integration -v
docker compose exec ollama ollama ps
```

The integration test embeds two synthetic strings and a query, checks matching
nonempty dimensions and finite values, and polls `/api/ps` for up to 10 seconds
after each request to verify the model unloads. Missing Ollama or a missing model
fails the selected test. Run it without concurrent embedding requests so model
unloading can be observed reliably. Use `docker compose stats` in another terminal
to observe memory during embedding; retain the existing 2 GB Ollama limit.

To run the existing Ruff checks in the test image:

```bash
docker compose --profile test run --rm --no-deps test \
  .venv/bin/ruff check app/embeddings tests/test_embeddings.py
docker compose --profile test run --rm --no-deps test \
  .venv/bin/ruff format --check app/embeddings tests/test_embeddings.py
```

An existing host development environment can also run the isolated checks with
`uv run --frozen --no-sync pytest -m "not integration" -v`. Keep real embedding
verification inside Compose; Ollama has no published host port.

## Markdown ingestion

Indexing is explicit; starting the API never scans documents. After rebuilding
`api`, inspect extraction without accessing PostgreSQL or Ollama:

```bash
docker compose exec api .venv/bin/python -m app.ingestion.sync --path /documents --inspect
```

Then index the same read-only mount:

```bash
docker compose exec api .venv/bin/python -m app.ingestion.sync --path /documents
```

`--path` defaults to `DOCUMENTS_DIR` (`/documents`). Inspection prints one JSON
object per chunk to stdout. Progress totals and concise per-file errors go to
stderr; any failure returns exit status 1 while other files continue. Indexing
requires `DATABASE_URL` and the configured local embedding model. No models are
downloaded by ingestion. Substitute `docker-compose` on standalone installations.

The loader accepts UTF-8 `.md` files recursively, skips symlinks, preserves text
formatting, and records ATX/Setext heading ancestry. Optional YAML frontmatter must
be a mapping. Dates become ISO strings; user fields remain nested under
`frontmatter`, separate from generated citation fields. The synthetic examples
illustrate `source`, `valid_from`, and `expires_on`. Expired evidence is retained
with its dates; ingestion does not certify validity or filter expired passages.

One database index represents one logical document root. Paths are relative to
that root, so changing the host/container mount location preserves IDs. Chunks
are bounded to 600 characters with up to 80 characters of overlap within a
section. IDs and chunk order are deterministic. Model and chunking configuration
are recorded and incompatible runs fail. Each file commits atomically after
embedding, and unchanged files are skipped. Changed indexed files are rejected;
replacement, rename handling, and deletion detection belong to Phase 5. Do not
reuse this index for an unrelated root or delete runtime data to resolve errors.

The storage tables are `ragamuffin_index`, `ragamuffin_documents`, and
`ragamuffin_chunks`; chunks contain text, pgvector embeddings, and JSONB citation
metadata. Embedding requests use batches of at most 16 chunks and the existing
`keep_alive=0` client. A file's extracted chunks and vectors remain in memory
until its transaction; unusually large files may exceed the API memory limit.

For validation, build the test image, start the separate PostgreSQL service, then
run the tests. Its database uses temporary storage and never mounts `data/`:

```bash
docker compose --profile test build test
docker compose --profile test up -d postgres-test ollama
docker compose --profile test run --rm --no-deps test
docker compose --profile test run --rm --no-deps test \
  .venv/bin/python -m pytest -m integration -v
docker compose --profile test run --rm --no-deps test \
  .venv/bin/ruff check app/ingestion tests/test_ingestion.py tests/test_ingestion_integration.py
docker compose --profile test run --rm --no-deps test \
  .venv/bin/ruff format --check app/ingestion tests/test_ingestion.py tests/test_ingestion_integration.py
```

Wait for the test database health check before integration tests. They require
`TEST_DATABASE_URL` pointing at a disposable database and an existing Ollama model;
unavailable services fail explicitly. Integration runs create isolated schemas
in that test database. Keep other embedding requests idle to verify unloading.
Use `docker compose stats` during integration tests to observe memory.
