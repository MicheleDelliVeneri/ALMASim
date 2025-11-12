#!/bin/bash
# Backend startup script
# This ensures the almasim package is available in the Python path

cd "$(dirname "$0")"
# Add parent directory to Python path so almasim package can be imported
export PYTHONPATH="${PYTHONPATH}:$(dirname "$(pwd)")"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

