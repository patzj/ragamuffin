# Ragamuffin

A lightweight, fully local RAG service for Markdown documents, built with FastAPI, PostgreSQL/pgvector, and Ollama.

The infrastructure and health endpoint are ready. Document indexing and retrieval are still in development.

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
