# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

WORKDIR /app

# System deps: none required beyond what pip needs for chromadb's sqlite
# bindings, which ship as wheels — this stays intentionally minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/chroma_db /app/logs

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# --- API image -------------------------------------------------------------
FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
# Ingest the seed knowledge base at image build/start time so a fresh
# container isn't serving an empty vector store on its very first request.
CMD ["sh", "-c", "python scripts/ingest_data.py && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"]

# --- Streamlit UI image -----------------------------------------------------
FROM base AS ui
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
