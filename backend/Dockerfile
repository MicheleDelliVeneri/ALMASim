FROM --platform=linux/amd64 python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/backend \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md /app/
COPY src/ /app/src/
COPY backend/ /app/backend/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-packages

ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p /app/outputs /app/data/query_results /tmp/download /root/.casa/data \
    && python -c "import almasim; print(f'almasim installed: {almasim.__file__}')"

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
