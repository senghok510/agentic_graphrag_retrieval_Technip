"""Branch: Aggregation / List Query (LightRAG local + global).

Mermaid path ``AQ → {EL1 → LG, HK → SR}``:
  * Online Entity Linking (EL1) → Local Graph Retrieval, 1-hop neighborhood (LG)
  * High-level Keyword Extraction (HK) → Semantic Relation Retrieval (SR)

Verbatim port of ``run_lightrag_pipeline_cross_encoder`` in evaluate_lightrag.py,
retargeted to the pipeline modules. Emits ranked chunk rows for Candidate Context
Fusion plus a text ``context`` block for the answer stage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .clients import embed_texts_large_model, _l2
from .entity_linking import link_entities
from .fusion import rerank_relations, _dedup_by_key
from .graph_retrieval import (
    _local_candidate_relations,
    _global_search,
    _relation_window_chunks,
    _attach_graph_weight,
)
from .clients import neo4j_driver
from .answer_context import clean_whitespace
from .trace import Stage, noop
from .views import relation_view, entity_view

logger = logging.getLogger("agent_flow.pipeline.lightrag")


def _build_answer_context(relations: List[Dict[str, Any]]) -> str:
    blocks = []
    for r in relations:
        subject = r.get("subject", "Unknown")
        obj = r.get("object", "Unknown")
        label = r.get("label", "related_to")
        description = r.get("description", "")
        lines = [f"Relation: {subject} --{label}--> {obj}", f"Description: {description}"]
        for src in r.get("sources") or []:
            file_name = src.get("file_name", "Unknown")
            text = clean_whitespace(src.get("text", ""))
            lines.append(f"Source clause [{file_name}]:\n{text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(f"<context>\n{b}\n</context>" for b in blocks)


def _window_chunks_to_rows(chunks: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Normalise ASSERTS window chunks (prev/match/next merged) → fusion rows."""
    rows = []
    for ch in chunks:
        parts = []
        if ch.get("prev_text"):
            parts.append(ch["prev_text"])
        parts.append(ch.get("text", ""))
        if ch.get("next_text"):
            parts.append(ch["next_text"])
        rows.append({
            "chunk_id": ch.get("chunk_id", ""),
            "content": "\n".join(p for p in parts if p),
            "file_name": ch.get("file_name", "Unknown"),
            "page_number": ch.get("page_number", "N/A"),
            "score": float(ch.get("rel_support", 0) or 0),
            "source": source,
        })
    return rows


def _cosine_rerank_local(question: str, local_rels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank local 1-hop relations by cosine similarity of their embedding to the question."""
    if not local_rels:
        return local_rels
    ids = [c["relation_id"] for c in local_rels]
    with neo4j_driver().session() as s:
        emap = {r["rid"]: r["emb"] for r in s.run(
            "MATCH (r:Relation) WHERE r.relationId IN $ids "
            "RETURN r.relationId AS rid, r.embedding AS emb", {"ids": ids}).data()}
    q_vec = _l2(embed_texts_large_model([question])[0])
    for c in local_rels:
        v = emap.get(c["relation_id"])
        c["cosine_score"] = float(np.dot(q_vec, _l2(v))) if v is not None else 0.0
    return sorted(local_rels, key=lambda c: c["cosine_score"], reverse=True)


def run_single_hop(analysis: dict, domains: Optional[List[str]] = None, emit=None,
                   candidate_pool: int = 500, local_top_n: int = 10,
                   max_chunks_local: int = 5) -> Dict[str, Any]:
    """Route 2 — Single-Hop Graph Lookup (LightRAG *local* search only).

    Entity linking → one-hop neighbours of the linked entities (domain-scoped).
    No global channel.
    """
    emit = emit or noop
    question = analysis["question"]
    entity_hints = [h.strip() for h in analysis.get("entity_hints", []) if h and h.strip()]
    category_hints = set(analysis.get("category_hints", []))

    with Stage(emit, "el1", "Online Entity Linking", "link_entities", "graph") as st:
        linked = link_entities(question, entity_hints=entity_hints, category_hints=category_hints)
        linked_keys = [c["entity_key"] for c in linked]
        st.set(linked=[c.get("entity_name") for c in linked], linked_count=len(linked_keys))
        st.attach("entities", "Linked entities", entity_view(linked))

    with Stage(emit, "single_hop", "Single-Hop Local Graph Retrieval (1-hop)",
               "_local_candidate_relations", "graph") as st:
        local_rels = _dedup_by_key(
            _local_candidate_relations(linked_keys, pool=candidate_pool, domains=domains), "relation_id")
        for r in local_rels:
            r["channel"] = "local"
        local_rels = local_rels[:local_top_n]
        local_rels = _attach_graph_weight(list(local_rels))
        st.set(domains=domains or "full", relations=len(local_rels))
        st.attach("relations", "One-hop relations", relation_view(local_rels))

    chunks_local = _dedup_by_key(
        _relation_window_chunks([r["relation_id"] for r in local_rels], max_chunks_local), "chunk_id")
    return {
        "local_rels": local_rels,
        "global_rels": [],
        "chunks": _window_chunks_to_rows(chunks_local, "graph_single_hop"),
        "context": _build_answer_context(local_rels),
        "linked_entities": [c.get("entity_name") for c in linked],
    }


def run_aggregation(analysis: dict, domains: Optional[List[str]] = None, emit=None,
                    candidate_pool: int = 2000, global_top_k: int = 100,
                    max_chunks_local: int = 30, max_chunks_global: int = 40,
                    local_top_n: int = 30, global_top_n: int = 80) -> Dict[str, Any]:
    """Route 3 — Aggregation / Summarization (LightRAG dual-level: local + global),
    with graph retrieval restricted to the predicted ``domains`` (None = full graph)."""
    emit = emit or noop
    question = analysis["question"]
    entity_hints = [h.strip() for h in analysis.get("entity_hints", []) if h and h.strip()]
    category_hints = set(analysis.get("category_hints", []))
    high_level_keywords = set(analysis.get("high_level_keywords", []))

    # EL1 → LG: linked entities → their 1-hop relations, cosine-reranked.
    with Stage(emit, "el1", "Online Entity Linking", "link_entities", "graph") as st:
        linked = link_entities(question, entity_hints=entity_hints, category_hints=category_hints)
        linked_keys = [c["entity_key"] for c in linked]
        st.set(linked=[c.get("entity_name") for c in linked], linked_count=len(linked_keys))
        st.attach("entities", "Linked entities", entity_view(linked))

    with Stage(emit, "local_graph", "Local Graph Retrieval (1-hop neighborhood)",
               "_local_candidate_relations", "graph") as st:
        local_rels = _dedup_by_key(
            _local_candidate_relations(linked_keys, pool=candidate_pool, domains=domains), "relation_id")
        for r in local_rels:
            r["channel"] = "local"
        local_rels = _cosine_rerank_local(question, local_rels)
        st.set(domains=domains or "full", relations=len(local_rels))
        st.attach("relations", "Local 1-hop relations (top)", relation_view(local_rels))

    # HK → SR: high-level theme keywords → relation vector search, cross-encoder reranked.
    with Stage(emit, "semantic_relation", "High-level Keywords → Semantic Relation Retrieval",
               "_global_search", "graph") as st:
        global_rels = _dedup_by_key(
            _global_search(high_level_keywords, top_k=global_top_k, domains=domains), "relation_id")
        # global_rels = rerank_relations(question, global_rels, top_k=60)
        for r in global_rels:
            r["channel"] = "global"
        st.set(high_level_keywords=list(high_level_keywords), relations=len(global_rels))
        st.attach("relations", "Semantic (global) relations", relation_view(global_rels))

    # cross-channel dedup so a relation surfaced locally does not repeat globally
    local_rel_ids = {r["relation_id"] for r in local_rels}
    global_rels = [r for r in global_rels if r["relation_id"] not in local_rel_ids]

    local_rels = local_rels[:local_top_n]
    global_rels = global_rels[:global_top_n]
    global_rels = _attach_graph_weight(list(global_rels))
    local_rels = _attach_graph_weight(list(local_rels))

    chunks_local = _dedup_by_key(_relation_window_chunks([r["relation_id"] for r in local_rels], max_chunks_local), "chunk_id")
    chunks_global = _dedup_by_key(_relation_window_chunks([r["relation_id"] for r in global_rels], max_chunks_global), "chunk_id")
    local_chunk_ids = {c["chunk_id"] for c in chunks_local}
    chunks_global = [c for c in chunks_global if c["chunk_id"] not in local_chunk_ids]

    local_context = _build_answer_context(local_rels)
    global_context = _build_answer_context(global_rels)

    chunk_rows = (_window_chunks_to_rows(chunks_local, "lightrag_local")
                  + _window_chunks_to_rows(chunks_global, "lightrag_global"))

    return {
        "local_rels": local_rels,
        "global_rels": global_rels,
        "chunks": chunk_rows,
        "context": "\n".join([global_context, local_context]),
        "linked_entities": [c.get("entity_name") for c in linked],
    }
