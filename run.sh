#!/usr/bin/env bash
# Runs both the API and the Streamlit UI together for local development.
# API runs in the background; Ctrl+C stops both (see the trap below).
set -euo pipefail
cd "$(dirname "$0")"

./run_api.sh &
API_PID=$!

cleanup() {
    echo ""
    echo "Stopping API (pid $API_PID)..."
    kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for the API to become healthy..."
API_PORT="${API_PORT:-8000}"
for _ in $(seq 1 30); do
    if curl -sf "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
        echo "API is up."
        break
    fi
    sleep 1
done

./run_ui.sh
