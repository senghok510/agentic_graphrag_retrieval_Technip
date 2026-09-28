"""FastAPI app exposing the adaptive retrieval pipeline.

Run it with:

    uvicorn pipeline_app:app --port 7060 --env-file ../.env --reload

Then point the UI at http://localhost:7060 (CHAT_API_URL / NEXT_PUBLIC_API_URL).
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import pipeline_router

logger = logging.getLogger("agent_flow.pipeline_app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the cross-encoder once at startup so the first question isn't slowed
    # by the model load. Set RERANKER_WARMUP=0 to skip (e.g. for a fast boot).
    if os.getenv("RERANKER_WARMUP", "1") != "0":
        from src.pipeline.fusion import warmup_reranker

        try:
            logger.info("Warming up cross-encoder at startup…")
            await asyncio.to_thread(warmup_reranker)
            logger.info("Cross-encoder warm-up complete")
        except Exception as exc:  # don't block startup if the model can't load
            logger.warning("Cross-encoder warm-up failed (will lazy-load): %s", exc)
    try:
        yield
    finally:
        from src.services.neo4j_service import close_neo4j_driver

        close_neo4j_driver()


def create_app() -> FastAPI:
    """Build the API application without performing import-time I/O."""

    application = FastAPI(
        title="Tender Pipeline API",
        version="0.1.0",
        lifespan=lifespan,
    )
    allowed_origins = [
        origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(pipeline_router.router)

    @application.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Tender Pipeline API",
            "docs": "/docs",
            "pipeline": "/pipeline/stream",
        }

    return application


app = create_app()
