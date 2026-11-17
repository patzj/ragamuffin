FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
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
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
