#!/usr/bin/env bash
# Runs the Streamlit frontend locally. Expects the API to already be
# running (see run_api.sh) — the sidebar will show a clear warning if it
# isn't reachable, but every page still renders.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "No virtualenv detected — creating ./venv (first run only)..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip -q
    ./venv/bin/pip install -r requirements.txt -q
fi

STREAMLIT_BIN="streamlit"
if [ -d "venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    STREAMLIT_BIN="venv/bin/streamlit"
fi

if [ ! -f ".env" ]; then
    cp .env.example .env
fi

export API_BASE_URL="${API_BASE_URL:-http://localhost:${API_PORT:-8000}}"
echo "Starting Streamlit UI on http://localhost:8501 (backend expected at ${API_BASE_URL}) ..."
exec "$STREAMLIT_BIN" run app.py --server.address=0.0.0.0 --server.port=8501
