"""Bounded "inspection" views for pipeline stage payloads.

Turn the internal retrieval structures (chunk rows, relation rows, entities)
into compact, JSON-safe lists that a UI can render when a user clicks a stage.
Everything is capped in count and truncated in length so the SSE frames stay
small even though the pipeline handles thousands of rows internally.
"""

from __future__ import annotations

from typing import Any


def _clip(text: Any, max_chars: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= max_chars else s[:max_chars].rstrip() + "…"


def chunk_view(
    rows: list[dict[str, Any]], n: int = 15, max_chars: int = 700
) -> list[dict[str, Any]]:
    """Chunk rows → [{chunk_id, file_name, page_number, score, source, text}]."""
    out = []
    for r in (rows or [])[:n]:
        score = r.get("ce_score", r.get("rrf_score", r.get("score", r.get("azure_score"))))
        out.append(
            {
                "chunk_id": r.get("chunk_id") or r.get("ChunkID") or "",
                "file_name": r.get("file_name", "Unknown"),
                "page_number": str(r.get("page_number", "N/A")),
                "score": round(float(score), 4) if isinstance(score, (int, float)) else None,
                "source": r.get("source")
                or (", ".join(r.get("sources", [])) if r.get("sources") else ""),
                "text": _clip(r.get("content") or r.get("text") or r.get("page_chunk"), max_chars),
            }
        )
    return out


def relation_view(
    rels: list[dict[str, Any]], n: int = 30, max_chars: int = 300
) -> list[dict[str, Any]]:
    """Relation rows → [{subject, label, object, description, score, channel}]."""
    out = []
    for r in (rels or [])[:n]:
        score = r.get(
            "ce_score",
            r.get(
                "final_score",
                r.get(
                    "cosine_score",
                    r.get("ppr_relation_score", r.get("vector_score", r.get("weight"))),
                ),
            ),
        )
        out.append(
            {
                "subject": r.get("subject", "?"),
                "label": r.get("label") or r.get("relationship_label") or "related_to",
                "object": r.get("object", "?"),
                "description": _clip(
                    r.get("description") or r.get("relationship_description"), max_chars
                ),
                "score": round(float(score), 4) if isinstance(score, (int, float)) else None,
                "channel": r.get("channel", ""),
            }
        )
    return out


def entity_view(entities: list[Any], n: int = 40) -> list[dict[str, Any]]:
    """Entities (dicts or bare names) → [{name, category, score}]."""
    out = []
    for e in (entities or [])[:n]:
        if isinstance(e, str):
            out.append({"name": e, "category": "", "score": None})
            continue
        score = e.get("ppr_score", e.get("seed_score", e.get("score")))
        out.append(
            {
                "name": e.get("entity_name") or e.get("name") or "?",
                "category": e.get("canonical_category", e.get("category", "")),
                "score": round(float(score), 6) if isinstance(score, (int, float)) else None,
            }
        )
    return out


def scored_entity_view(
    entity_scores: dict[str, float], entity_meta: dict[str, dict], n: int = 30
) -> list[dict[str, Any]]:
    """PPR entity_scores {key: score} + meta → ranked [{name, category, score}]."""
    ranked = sorted(entity_scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
    out = []
    for key, score in ranked:
        meta = entity_meta.get(key, {})
        out.append(
            {
                "name": meta.get("entity_name", key),
                "category": meta.get("canonical_category", ""),
                "score": round(float(score), 6),
            }
        )
    return out
