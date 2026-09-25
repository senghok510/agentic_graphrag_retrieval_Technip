"""Shared retrieval primitives.

Covers the mermaid leaves that read raw evidence:
  * Domain-aware Hybrid Search  → ``hybrid_search`` (Azure AI Search, BM25 + vector)
  * Semantic Retrieval (SEM)    → ``vector_seed`` (chunk vectors, normalised rows)
  * Cypher Template Retrieval   → ``cypher_template_retrieval`` (KG full-text on chunks)

All chunk rows are normalised to the pipeline's common shape:
``{chunk_id, content, file_name, page_number, score, source}``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .clients import TARGET_ITB_ID, neo4j_driver
from ..services.azure_search import azure_search_tool

logger = logging.getLogger("agent_flow.pipeline.retrieval")


def hybrid_search(query: str, tender_id: Optional[str] = None, top: int = 50,
                  vector_query_text: Optional[str] = None) -> List[Dict[str, Any]]:
    """Azure AI Search hybrid (semantic + vector) query → raw tool rows.

    ``query`` drives BM25/semantic; ``vector_query_text`` (HyDE) drives the
    vector leg. Handles the LangChain ``@tool`` wrapper transparently.
    """
    payload = {
        "query": query,
        "tender_id": tender_id,
        "top": top,
        "vector_query_text": vector_query_text or query,
    }
    try:
        if hasattr(azure_search_tool, "invoke"):
            return azure_search_tool.invoke(payload)
        return azure_search_tool(**payload)
    except Exception as exc:  # pragma: no cover - depends on live search
        logger.error("hybrid_search error: %s", exc)
        return []


def hybrid_rag(analysis: dict, tender_id: Optional[str] = None, top: int = 50) -> List[Dict[str, Any]]:
    """Hybrid RAG evidence branch (v2) — BM25 + dense vector over the FULL Azure index.

    Per the spec this is never restricted by domain and runs for every route
    (the only source for Textual Factoid; parallel to graph retrieval otherwise).
    Returns normalised chunk rows tagged ``hybrid_rag``.
    """
    rows = hybrid_search(
        query=analysis.get("expanded_query") or analysis.get("question", ""),
        tender_id=tender_id,
        top=top,
        vector_query_text=analysis.get("hyde_doc"),
    )
    return _normalise_search_rows(rows, source="hybrid_rag")


def _normalise_search_rows(rows: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append({
            "chunk_id": row.get("CurrentChunkID") or row.get("chunk_id") or "",
            "content": row.get("page_chunk", "") or row.get("content", ""),
            "file_name": row.get("file_name", "Unknown"),
            "page_number": row.get("page_number", "N/A"),
            "score": float(row.get("reranker_score", 0.0) or 0.0),
            "source": source,
        })
    return out


def vector_seed(analysis: dict, tender_id: str = TARGET_ITB_ID, top: int = 50) -> List[Dict[str, Any]]:
    """Semantic Retrieval seed: expanded query (BM25) + HyDE (vector).

    Reproduces the notebook ``vector_seed`` including its score-threshold gate.
    """
    raw = hybrid_search(
        query=analysis.get("expanded_query") or analysis.get("question", ""),
        tender_id=tender_id,
        top=top,
        vector_query_text=analysis.get("hyde_doc"),
    )
    max_score = max((row.get("reranker_score", 0.0) for row in raw), default=0.0)
    current_threshold = 1 if max_score > 0.1 else 0.0
    filtered = [row for row in raw if row.get("reranker_score", 0.0) >= current_threshold]

    out = []
    for row in filtered:
        out.append({
            "chunk_id": row.get("CurrentChunkID") or row.get("chunk_id") or "",
            "content": row.get("page_chunk", ""),
            "file_name": row.get("file_name", "Unknown"),
            "page_number": row.get("page_number", "N/A"),
            "azure_score": float(row.get("reranker_score", 0.0)),
            "score": float(row.get("reranker_score", 0.0)),
            "source": "vector_seed",
            "contributions": {"vector_seed": float(row.get("reranker_score", 0.0))},
        })
    logger.info("[vector_seed] kept %d chunks (threshold=%s)", len(out), current_threshold)
    return out


def semantic_retrieval(analysis: dict, tender_id: Optional[str] = None, top: int = 20) -> List[Dict[str, Any]]:
    """Simple-query Semantic Retrieval (SEM leaf) → normalised chunk rows."""
    rows = hybrid_search(
        query=analysis.get("expanded_query") or analysis.get("question", ""),
        tender_id=tender_id,
        top=top,
        vector_query_text=analysis.get("hyde_doc"),
    )
    return _normalise_search_rows(rows, source="semantic_retrieval")


# ── Cypher Template Retrieval (Simple Query branch) ──────────────────────────

# Full-text over chunk text — a fast, deterministic KG lookup for atomic facts.
_CHUNK_FT_CYPHER = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q)
YIELD node AS ch, score
OPTIONAL MATCH (ch)-[:PART_OF]->(d:Document)
WITH ch, d, score
"""
_CHUNK_FT_TAIL = """
RETURN ch.ChunkID AS chunk_id,
       coalesce(ch.text, '') AS content,
       coalesce(ch.fileName, d.fileName, d.docURL, 'Unknown') AS file_name,
       coalesce(toString(ch.pageNumber), toString(ch.sequenceNo), 'N/A') AS page_number,
       score
ORDER BY score DESC
LIMIT $top
"""


def _build_chunk_lucene(entity_hints: List[str], keywords: List[str]) -> str:
    terms = []
    for value in list(entity_hints or []) + list(keywords or []):
        value = (value or "").strip()
        if not value:
            continue
        if " " in value:
            terms.append(f'"{value}"')
        else:
            terms.append(value)
    return " OR ".join(dict.fromkeys(terms))


def cypher_template_retrieval(analysis: dict, tender_id: Optional[str] = None,
                              top: int = 15) -> List[Dict[str, Any]]:
    """Cypher Template Retrieval (Simple Query leaf).

    Deterministic KG full-text lookup: entity hints + expansion keywords →
    ``chunk_text_ft`` → chunk rows. Optionally scoped to an ITB via the
    document link. Returns normalised chunk rows.
    """
    lucene = _build_chunk_lucene(
        analysis.get("entity_hints", []),
        analysis.get("expansion_keywords", []),
    )
    if not lucene:
        return []

    if tender_id:
        cypher = (
            _CHUNK_FT_CYPHER
            + "OPTIONAL MATCH (d)-[:BELONGS_TO]->(itb:ITB)\n"
            + "WITH ch, d, score WHERE itb IS NULL OR itb.ITB_ID = $tender_id\n"
            + _CHUNK_FT_TAIL
        )
        params = {"q": lucene, "top": top, "tender_id": tender_id}
    else:
        cypher = _CHUNK_FT_CYPHER + _CHUNK_FT_TAIL
        params = {"q": lucene, "top": top}

    try:
        with neo4j_driver().session() as s:
            rows = [r.data() for r in s.run(cypher, params)]
    except Exception as exc:  # pragma: no cover - depends on live DB / index
        logger.warning("cypher_template_retrieval error: %s", exc)
        return []

    for r in rows:
        r["source"] = "cypher_template"
    return rows
