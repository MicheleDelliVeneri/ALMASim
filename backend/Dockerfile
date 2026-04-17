FROM --platform=linux/amd64 python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/backend

WORKDIR /app

COPY pyproject.toml setup.py README.md /app/
COPY backend/requirements.txt /app/backend/requirements.txt
COPY src/ /app/src/
COPY backend/ /app/backend/

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && cd /app && pip install --no-cache-dir -e . \
    && grep -v '^-e' /app/backend/requirements.txt > /tmp/backend-deps.txt \
    && pip install --no-cache-dir -r /tmp/backend-deps.txt \
    && apt-get purge -y gcc g++ && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/outputs /app/data/query_results /tmp/download /root/.casa/data \
    && python -c "import almasim; print(f'almasim installed: {almasim.__file__}')"

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
