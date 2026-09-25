"""Server-Sent-Events streaming for the pipeline.

``stream_pipeline_sse`` runs the (blocking) pipeline in a worker thread and
forwards every stage event to the async caller as it happens, then a terminal
``final`` (or ``error``) event carrying the grounded answer. Each yielded string
is a ready-to-send SSE frame (``data: {json}\\n\\n``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from .orchestrator import run_pipeline

logger = logging.getLogger("agent_flow.pipeline.streaming")

_DONE = object()


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


async def stream_pipeline_sse(question: str, tender_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue" = asyncio.Queue()

    def emit(event: dict):
        # Called from the worker thread → hop back onto the event loop safely.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _run():
        try:
            result = run_pipeline(question, tender_id=tender_id, emit=emit)
            answer = result["answer"]
            debug = result.get("debug", {})
            emit({
                "type": "final",
                "answer": answer.answer,
                "references": [r.model_dump() if hasattr(r, "model_dump") else r for r in answer.references],
                "confidence": answer.confidence_score,
                "route": result.get("route"),                 # retrieval need
                "domain_scope": debug.get("domain_scope"),     # single | multi | full
                "graph_domains": debug.get("graph_domains"),   # scoped domains (or null)
                "debug": debug,
            })
        except Exception as exc:  # surface failures to the UI instead of hanging
            logger.exception("pipeline stream failed")
            emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    task = loop.run_in_executor(None, _run)

    # Open the stream so the client renders immediately.
    yield _sse({"type": "open", "question": question, "tender_id": tender_id})
    try:
        while True:
            event = await queue.get()
            if event is _DONE:
                break
            yield _sse(event)
    finally:
        await task
    yield "data: [DONE]\n\n"
