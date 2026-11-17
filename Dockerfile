FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS base
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

FROM base AS test
RUN uv sync --frozen --group dev
COPY app ./app
COPY tests ./tests
CMD [".venv/bin/python", "-m", "pytest", "-m", "not integration", "-v"]

FROM base AS runtime
RUN uv sync --frozen --no-dev
COPY app ./app
EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.main:app", "--no-access-log", "--host", "0.0.0.0", "--port", "8000"]
