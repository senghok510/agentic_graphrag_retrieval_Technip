"""Compatibility entry point for the LangGraph-based adaptive pipeline.

Callers keep importing :func:`run_pipeline` from this module. The actual
orchestration lives in ``langgraph_pipeline``; the result contract is unchanged.
"""

from __future__ import annotations

from .langgraph_pipeline import (
    FINAL_CONTEXT_CHUNKS,
    GRAPH_BRANCH,
    HYBRID_TOP,
    RRF_POOL,
    run_langgraph_pipeline,
)
from .trace import EmitFn


def run_pipeline(
    question: str,
    tender_id: str | None = None,
    top: int = HYBRID_TOP,
    emit: EmitFn | None = None,
    disable_domain_filter: bool = True,
) -> dict:
    """Execute the adaptive retrieval workflow using the compiled LangGraph."""

    return run_langgraph_pipeline(
        question=question,
        tender_id=tender_id,
        top=top,
        emit=emit,
        disable_domain_filter=disable_domain_filter,
    )


__all__ = [
    "FINAL_CONTEXT_CHUNKS",
    "GRAPH_BRANCH",
    "HYBRID_TOP",
    "RRF_POOL",
    "run_pipeline",
]
