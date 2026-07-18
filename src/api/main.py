"""
FastAPI application entry point.

Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
(or `./run_api.sh`, which does exactly that with the right env loaded).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agents.rag_agent import get_retriever
from src.api.routes import router
from src.config import get_settings
from src.graph.workflow import get_graph_manager
from src.monitoring.logging_config import get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("api_startup", extra={"environment": settings.environment})

    # Warm both singletons at startup rather than on first request: the
    # first caller shouldn't pay for vector-store/checkpoint-DB
    # initialization latency, and any config problem (bad path, corrupt
    # DB file) surfaces at boot instead of on someone's first /plan call.
    get_retriever()
    manager = get_graph_manager()
    logger.info("api_ready")

    yield

    manager.close()
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI Traveller API",
        description="Multi-agent AI travel planning system — see /docs for interactive Swagger UI.",
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = settings.cors_origin_list()
    allow_all = "*" in origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else origins,
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", extra={"path": str(request.url), "error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(exc)})

    app.include_router(router)
    return app


app = create_app()
