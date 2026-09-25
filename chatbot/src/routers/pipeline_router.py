"""FastAPI router for the main_project_pipeline.mmd flow.

Endpoints:
  * ``POST /pipeline/ask``    — run the pipeline, return the final answer as JSON
  * ``POST /pipeline/stream`` — Server-Sent-Events: one frame per stage, then the answer
  * ``GET  /pipeline/health`` — liveness probe
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..pipeline.orchestrator import run_pipeline
from ..pipeline.streaming import stream_pipeline_sse

logger = logging.getLogger("agent_flow.pipeline_router")

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class AskRequest(BaseModel):
    question: str
    tender_id: Optional[str] = None


class Reference(BaseModel):
    file_name: str
    page_number: str


class AskResponse(BaseModel):
    answer: str
    references: List[Reference] = []
    confidence: float = 0.0
    route: Optional[str] = None
    strategy: Optional[str] = None
    debug: Dict[str, Any] = {}


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    import asyncio
    try:
        result = await asyncio.to_thread(run_pipeline, req.question, req.tender_id)
    except Exception as exc:
        logger.exception("pipeline /ask failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    answer = result["answer"]
    return AskResponse(
        answer=answer.answer,
        references=[{"file_name": r.file_name, "page_number": r.page_number} for r in answer.references],
        confidence=answer.confidence_score,
        route=result.get("route"),
        strategy=result.get("strategy"),
        debug=result.get("debug", {}),
    )


@router.post("/stream")
async def stream(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    return StreamingResponse(
        stream_pipeline_sse(req.question, req.tender_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so events arrive live
        },
    )
