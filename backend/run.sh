#!/bin/bash
# Backend startup script
# This ensures the almasim package is available in the Python path

cd "$(dirname "$0")"
# Add parent directory to Python path so almasim package can be imported
export PYTHONPATH="${PYTHONPATH}:$(dirname "$(pwd)")"
# The API has no authentication: listen on this machine only unless
# ALMASIM_HOST says otherwise.
uvicorn app.main:app --reload --host "${ALMASIM_HOST:-127.0.0.1}" --port 8000

