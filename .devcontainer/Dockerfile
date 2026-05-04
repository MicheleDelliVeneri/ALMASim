FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 casacore-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency sync in the workspace venv.
RUN pip install --no-cache-dir uv
