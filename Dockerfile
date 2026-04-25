FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
COPY src/ /app/src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group test

ENV PATH="/app/.venv/bin:$PATH"

COPY . .

CMD ["pytest", "tests"]
