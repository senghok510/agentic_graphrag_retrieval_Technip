"""Pydantic models used across the pipeline.

Re-exports the app's ``ResponseGeneration`` / ``ReferenceGeneration`` /
``ComplexQueryExpansion`` / ``SimpleQueryExpansion`` so the pipeline and the
FastAPI layer share one contract, and adds the graph-routing models the
notebooks defined inline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Reuse the shared application response contract.
from ..models.schemas import (  # noqa: F401
    ComplexQueryExpansion,
    ReferenceGeneration,
    ResponseGeneration,
    SimpleQueryExpansion,
)


class GraphComplexityAnalysis(BaseModel):
    """Combined complexity analysis + graph-routing hints (one LLM call)."""

    is_complex: bool = Field(description="Whether the query needs multi-hop reasoning")
    reasoning: str = Field(description="1-line explanation of why the query is complex or simple")
    sub_questions: list[str] = Field(
        default_factory=list, description="Sub-questions for complex queries (0-3)"
    )
    entity_hints: list[str] = Field(
        default_factory=list, description="Specific named things from the question"
    )
    category_hints: list[str] = Field(
        default_factory=list, description="Canonical entity categories (closed list)"
    )
    relation_hints: list[str] = Field(
        default_factory=list, description="snake_case relationship intent hints"
    )
    high_level_keywords: list[str] = Field(
        default_factory=list, description="Overarching theme keywords (snake_case)"
    )


class StrategyChoice(BaseModel):
    """Query-type classification for the graph branch (v1, kept for compatibility)."""

    strategy: Literal["factoid", "relational", "summarization"]
    reasoning: str


# ── v2: adaptive retrieval-need classification ───────────────────────────────

RetrievalNeedType = Literal["textual_factoid", "single_hop", "aggregation", "multi_hop"]


class RetrievalNeed(BaseModel):
    """Retrieval Need Classification (v2) — HOW to retrieve, independent of domain.

    * textual_factoid — answer is an explicit value in text → Hybrid RAG only.
    * single_hop      — one graph edge suffices, (s,p,*) or (*,p,o) → LightRAG local.
    * aggregation     — collect/summarize many related facts → Dual-Level.
    * multi_hop       — discover connections / traverse many edges, (s,*,o) → Hub-aware PPR.
    """

    need: RetrievalNeedType = Field(description="The primary retrieval algorithm to use")
    query_form: str = Field(default="none", description="(s,p,*) | (*,p,o) | (s,*,o) | none")
    reasoning: str = Field(default="", description="one-line justification")


class GraphDomainPrediction(BaseModel):
    """Graph Domain Prediction (v2) — WHERE the graph searches. Never affects Hybrid RAG."""

    scope: Literal["single", "multi", "full"] = Field(
        description="single/multi domain, or full graph (GENERAL)"
    )
    domains: list[str] = Field(
        default_factory=list, description="Exact domain names (empty when scope=full)"
    )
    reasoning: str = Field(default="", description="one-line justification")
