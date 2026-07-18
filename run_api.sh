#!/usr/bin/env bash
# Runs the FastAPI backend locally (no Docker). Assumes a virtualenv with
# requirements.txt installed is already active, or falls back to creating
# one at ./venv on first run.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "No virtualenv detected — creating ./venv (first run only)..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip -q
    ./venv/bin/pip install -r requirements.txt -q
fi

PYTHON_BIN="python3"
if [ -d "venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    PYTHON_BIN="venv/bin/python"
fi

if [ ! -f ".env" ]; then
    echo "No .env found — copying .env.example. Add your API keys before expecting AI-generated plans."
    cp .env.example .env
fi

echo "Ingesting seed knowledge base (safe to run repeatedly)..."
"$PYTHON_BIN" scripts/ingest_data.py

echo "Starting API on http://localhost:${API_PORT:-8000} ..."
exec "$PYTHON_BIN" -m uvicorn src.api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --reload
