<p align="center">
  <img src="logo.png" alt="Ragamuffin logo" width="280">
</p>

# Ragamuffin

A lightweight, fully local RAG service for Markdown documents, built with FastAPI, PostgreSQL/pgvector, and Ollama.

The infrastructure, health endpoint, and local embedding client are ready. Document indexing and retrieval are still in development.

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
