FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/backend

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for better layer caching)
COPY pyproject.toml setup.py README.md /app/
COPY backend/requirements.txt /app/backend/requirements.txt

# Install Python build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy source code
COPY src/ /app/src/
COPY backend/ /app/backend/

# Install almasim package first (with src layout)
# The package is in src/almasim, setup.py has package_dir={"": "src"}
RUN cd /app && pip install --no-cache-dir -e . && \
    python -c "import almasim; print(f'almasim installed: {almasim.__file__}')"

# Install backend dependencies from requirements.txt
RUN grep -v '^-e' /app/backend/requirements.txt > /tmp/backend-deps.txt && \
    pip install --no-cache-dir -r /tmp/backend-deps.txt

# Remove build dependencies to reduce image size
RUN apt-get purge -y gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/outputs /app/data/query_results /tmp/download

WORKDIR /app/backend

# Verify almasim can be imported (should work after pip install -e)
RUN python -c "import almasim; print(f'almasim import verified: {almasim.__file__}')"

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
