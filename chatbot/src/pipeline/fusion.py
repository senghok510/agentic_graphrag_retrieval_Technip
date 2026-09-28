"""Stage: Candidate Context Fusion → Reciprocal Rank Fusion → Cross-Encoder Rerank.

Mermaid nodes ``CF`` → ``RRF`` → ``CE``. Every retrieval branch emits a ranked
list of chunk rows; these are fused by RRF into one candidate pool and then
reranked against the question by the cross-encoder (``BAAI/bge-reranker-large``),
exactly as the notebooks do for relations.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from .clients import _l2, _minmax, embed_texts_large_model, neo4j_driver

logger = logging.getLogger("agent_flow.pipeline.fusion")

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))

# Process-wide singleton. The pipeline runs inside worker threads (asyncio
# to_thread / run_in_executor), so construction is guarded by a lock with
# double-checked locking to guarantee the model is loaded EXACTLY once even when
# several requests race on the first call.
_RERANKER = None
_RERANKER_LOCK = threading.Lock()


def get_reranker():
    """Return the shared cross-encoder, loading it once (thread-safe, cached)."""
    global _RERANKER
    if _RERANKER is None:
        with _RERANKER_LOCK:
            if _RERANKER is None:  # re-check inside the lock
                from sentence_transformers import CrossEncoder

                logger.info("Loading cross-encoder '%s' (one-time)…", RERANKER_MODEL)
                _RERANKER = CrossEncoder(RERANKER_MODEL, max_length=RERANKER_MAX_LENGTH)
                logger.info("Cross-encoder loaded and cached process-wide")
    return _RERANKER


def warmup_reranker() -> None:
    """Eagerly load the cross-encoder (call at server startup to avoid the
    first-question latency spike). Safe to call multiple times — it is a no-op
    once loaded."""
    get_reranker()


def _dedup_by_key(items, key):
    seen, out = set(), []
    for item in items:
        k = item.get(key)
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


# ── Relation reranking (used inside the branch modules) ──────────────────────


def rerank_relations(
    question: str, relations: list[dict[str, Any]], top_k: int = 30, alpha: float = 1.0
) -> list[dict[str, Any]]:
    """Cross-encoder relevance of relations, blended with graph weight.

    final = alpha * cross_encoder_norm + (1 - alpha) * graph_weight_norm.
    """
    if not relations:
        return []
    reranker = get_reranker()
    pairs = [
        (
            question,
            f"{r.get('subject', '')} --{r.get('label', '')}--> {r.get('object', '')}. {r.get('description', '')}",
        )
        for r in relations
    ]
    ce = reranker.predict(pairs, batch_size=32)
    ce_n = _minmax(ce)
    w_n = _minmax([r.get("weight", 0.0) for r in relations])
    for r, c, cn, wn in zip(relations, ce, ce_n, w_n, strict=True):
        r["ce_score"] = float(c)
        r["final_score"] = float(alpha * cn + (1 - alpha) * wn)
    return sorted(relations, key=lambda r: r["final_score"], reverse=True)[:top_k]


def cosine_rerank_relations(
    question: str, relations: list[dict[str, Any]], top_k: int = 30
) -> list[dict[str, Any]]:
    """Bi-encoder cosine rerank of relations against the question (fast path)."""
    if not relations:
        return []
    ids = [c["relation_id"] for c in relations]
    with neo4j_driver().session() as s:
        emap = {
            r["rid"]: r["emb"]
            for r in s.run(
                "MATCH (r:Relation) WHERE r.relationId IN $ids "
                "RETURN r.relationId AS rid, r.embedding AS emb",
                {"ids": ids},
            ).data()
        }
    q_vec = _l2(embed_texts_large_model([question])[0])
    for c in relations:
        v = emap.get(c["relation_id"])
        c["cosine_score"] = float(np.dot(q_vec, _l2(v))) if v is not None else 0.0
    return sorted(relations, key=lambda c: c["cosine_score"], reverse=True)[:top_k]


# ── Candidate context fusion (chunk level) ───────────────────────────────────


def _chunk_key(row: dict[str, Any]) -> str:
    return (
        row.get("chunk_id")
        or row.get("ChunkID")
        or f"{row.get('file_name')}|{row.get('page_number')}"
    )


def _chunk_content(row: dict[str, Any]) -> str:
    return row.get("content") or row.get("text") or row.get("page_chunk") or ""


def reciprocal_rank_fusion(
    chunk_lists: list[list[dict[str, Any]]], k: int = 60, top_k: int = 60
) -> list[dict[str, Any]]:
    """Candidate Context Fusion + RRF.

    ``chunk_lists`` is one already-ranked chunk list per retrieval branch. Each
    chunk's fused score is ``sum(1 / (k + rank))`` across the lists it appears
    in; ties keep the earliest/highest-ranked metadata. Returns the fused,
    de-duplicated candidate pool (normalised chunk rows).
    """
    fused: dict[str, dict[str, Any]] = {}
    for chunk_list in chunk_lists:
        for rank, row in enumerate(chunk_list or [], start=1):
            key = _chunk_key(row)
            if not key:
                continue
            rec = fused.get(key)
            if rec is None:
                rec = {
                    "chunk_id": row.get("chunk_id") or row.get("ChunkID") or "",
                    "content": _chunk_content(row),
                    "file_name": row.get("file_name", "Unknown"),
                    "page_number": row.get("page_number", "N/A"),
                    "sources": [],
                    "rrf_score": 0.0,
                }
                fused[key] = rec
            rec["rrf_score"] += 1.0 / (k + rank)
            src = row.get("source")
            if src and src not in rec["sources"]:
                rec["sources"].append(src)
            if not rec["content"]:
                rec["content"] = _chunk_content(row)
    ranked = sorted(fused.values(), key=lambda r: r["rrf_score"], reverse=True)
    for r in ranked:
        r["score"] = r["rrf_score"]
    return ranked[:top_k]


def rerank_chunks(
    question: str, chunks: list[dict[str, Any]], top_k: int = 20, snippet_chars: int = 1200
) -> list[dict[str, Any]]:
    """Cross-Encoder Reranking of the fused candidate chunks against the question."""
    chunks = [c for c in chunks if _chunk_content(c).strip()]
    if not chunks:
        return []
    reranker = get_reranker()
    pairs = [(question, _chunk_content(c)) for c in chunks]
    ce = reranker.predict(pairs, batch_size=24)
    for c, s in zip(chunks, ce, strict=True):
        c["ce_score"] = float(s)
    return sorted(chunks, key=lambda c: c["ce_score"], reverse=True)[:top_k]
