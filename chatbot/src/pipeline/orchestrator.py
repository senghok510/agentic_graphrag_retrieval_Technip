"""Adaptive retrieval pipeline orchestrator (v2 — main_project_pipeline_v2.mmd).

The architecture separates two INDEPENDENT decisions made from the query:

    Query Understanding
        ├── Retrieval Need Classification   (HOW to retrieve)
        │      textual_factoid | single_hop | aggregation | multi_hop
        └── Graph Domain Prediction         (WHERE the graph searches)
               single | multi | full graph      ← scopes GRAPH retrieval ONLY

    Route by retrieval need:
        textual_factoid → Hybrid RAG only
        single_hop      → Local Graph Retrieval          ┐  (domain-scoped)
        aggregation     → Dual-Level Retrieval           ├─ ∥ Hybrid RAG (full index)
        multi_hop       → Hub-aware PPR                   ┘

    Evidence Fusion + Deduplication → Cross-Encoder Reranking → LLM Answer

Key invariants from the spec:
  * Graph Domain Prediction NEVER restricts Hybrid RAG — Hybrid RAG always
    searches the complete Azure AI Search index.
  * Graph retrieval and Hybrid RAG are complementary and run in PARALLEL
    (every route except Textual Factoid, which is Hybrid-only).

Every stage calls the optional ``emit`` callback so a UI can show, live, which
method is running. ``emit`` defaults to a no-op.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .clients import TARGET_ITB_ID
from .query_understanding import analyze_query, classify_retrieval_need
from .domain_routing import predict_graph_domains
from .retrieval import hybrid_rag
from .lightrag import run_single_hop, run_aggregation
from .ppr import run_multihop
from .fusion import reciprocal_rank_fusion, rerank_chunks
from .answer import generate_answer
from .trace import Stage as _Stage, noop as _noop, EmitFn
from .views import chunk_view, relation_view, entity_view

logger = logging.getLogger("agent_flow.pipeline.orchestrator")

RRF_POOL = 50
FINAL_CONTEXT_CHUNKS = 20
HYBRID_TOP = 15

# route → the graph branch that runs (Textual Factoid has none)
GRAPH_BRANCH = {
    "single_hop": "Local Graph Retrieval",
    "aggregation": "Dual-Level Retrieval",
    "multi_hop": "Hub-aware PPR",
}


def _run_graph_branch(need: str, analysis: dict, domains: Optional[List[str]],
                      tender_id: str, emit: EmitFn) -> Dict[str, Any]:
    """Dispatch to the graph retrieval strategy for the classified need (domain-scoped)."""
    if need == "single_hop":
        return run_single_hop(analysis, domains=domains, emit=emit)
    if need == "aggregation":
        return run_aggregation(analysis, domains=domains, emit=emit)
    if need == "multi_hop":
        return run_multihop(analysis, domains=domains, emit=emit, tender_id=tender_id)
    return {"chunks": [], "relations": [], "context": ""}


def run_pipeline(question: str, tender_id: str = None, top: int = HYBRID_TOP,
                 emit: Optional[EmitFn] = None, disable_domain_filter: bool = True) -> Dict[str, Any]:
    """Execute the full adaptive pipeline for one question.

    Returns ``{answer: ResponseGeneration, route, domains, analysis, debug}``.
    Pass ``emit`` to receive a live event per stage (for a streaming UI).
    Pass ``disable_domain_filter=True`` to force full-graph search on every route,
    bypassing Graph Domain Prediction's scoping entirely (for A/B evaluation —
    see scripts/domain_eval/).
    """
    emit = emit or _noop
    tender_id = tender_id or TARGET_ITB_ID
    t0 = time.time()

    # ── Query Understanding ──────────────────────────────────────────────────
    with _Stage(emit, "query_understanding", "Query Understanding", "analyze_query", "understanding") as st:
        analysis = analyze_query(question)
        st.set(is_complex=analysis.get("is_complex"),
               entity_hints=analysis.get("entity_hints", []),
               high_level_keywords=analysis.get("high_level_keywords", []),
               sub_questions=analysis.get("sub_questions", []))
        st.attach("json", "Query analysis", {
            "entity_hints": analysis.get("entity_hints", []),
            "category_hints": analysis.get("category_hints", []),
            "high_level_keywords": analysis.get("high_level_keywords", []),
            "sub_questions": analysis.get("sub_questions", []),
            "expanded_query": analysis.get("expanded_query", ""),
        })
        st.attach("text", "HyDE document (used for the vector query)", analysis.get("hyde_doc", ""))

    # ── Two independent decisions (conceptually parallel) ────────────────────
    with _Stage(emit, "retrieval_need", "Retrieval Need Classification", "classify_retrieval_need", "classification") as st:
        need_choice = classify_retrieval_need(question)
        need = need_choice.need
        st.set(need=need, query_form=need_choice.query_form, reasoning=need_choice.reasoning)

    with _Stage(emit, "domain_prediction", "Graph Domain Prediction", "predict_graph_domains", "classification") as st:
        prediction = predict_graph_domains(question, analysis)
        st.set(scope=prediction["scope"], domains=prediction["domains"], reasoning=prediction.get("reasoning", ""))
        st.attach("json", "Domain prediction", {
            "scope": prediction["scope"], "domains": prediction["domains"],
            "keyword_scores": prediction.get("keyword_scores", {}),
            "reasoning": prediction.get("reasoning", ""),
        })

    # Graph scope selection: full graph → no domain filter.
    graph_domains = None if prediction["scope"] == "full" else (prediction["domains"] or None)
    if disable_domain_filter:
        graph_domains = None
    with _Stage(emit, "graph_scope", "Graph Scope Selection", "graph_scope", "classification") as st:
        st.set(scope=prediction["scope"], domains=graph_domains or "full graph",
               graph_branch=GRAPH_BRANCH.get(need, "— (no graph, Hybrid only)"),
               filter_disabled=disable_domain_filter)

    # ── Select primary retrieval strategy ────────────────────────────────────
    candidate_lists: List[List[Dict[str, Any]]] = []
    graph_result: Dict[str, Any] = {}
    graph_context = ""

    if need == "textual_factoid":
        # Route 1 — Hybrid RAG only, no graph.
        with _Stage(emit, "hybrid_rag", "Hybrid RAG (BM25 + Dense, full index)", "hybrid_rag", "retrieval") as st:
            hybrid = hybrid_rag(analysis, tender_id=tender_id, top=top)
            st.set(chunks=len(hybrid), scope="full Azure index")
            st.attach("chunks", "Hybrid RAG chunks (full index)", chunk_view(hybrid))
        candidate_lists = [hybrid]
    else:
        # Routes 2/3/4 — graph retrieval ∥ Hybrid RAG (both run concurrently).
        def hybrid_task():
            with _Stage(emit, "hybrid_rag", "Hybrid RAG (BM25 + Dense, full index)", "hybrid_rag", "retrieval") as st:
                rows = hybrid_rag(analysis, tender_id=tender_id, top=top)
                st.set(chunks=len(rows), scope="full Azure index")
                st.attach("chunks", "Hybrid RAG chunks (full index)", chunk_view(rows))
                return rows

        def graph_task():
            with _Stage(emit, "graph_retrieval", f"Graph Retrieval — {GRAPH_BRANCH[need]}",
                        "_run_graph_branch", "graph") as st:
                res = _run_graph_branch(need, analysis, graph_domains, tender_id, emit)
                st.set(chunks=len(res.get("chunks", [])), domains=graph_domains or "full")
                rels = res.get("relations") or (res.get("local_rels", []) + res.get("global_rels", []))
                st.attach("entities", "Linked / seed entities", entity_view(
                    res.get("linked_entities") or res.get("seeds") or []))
                st.attach("relations", "Graph relations", relation_view(rels))
                st.attach("chunks", "Graph evidence chunks", chunk_view(res.get("chunks", [])))
                return res

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_hybrid = ex.submit(hybrid_task)
            f_graph = ex.submit(graph_task)
            hybrid = f_hybrid.result()
            graph_result = f_graph.result()

        candidate_lists = [graph_result.get("chunks", []), hybrid]
        graph_context = graph_result.get("context") or _relations_to_context(graph_result.get("relations", []))

    # ── Evidence Fusion + Deduplication (RRF) ────────────────────────────────
    with _Stage(emit, "evidence_fusion", "Evidence Fusion & Deduplication (RRF)",
                "reciprocal_rank_fusion", "fusion") as st:
        fused = reciprocal_rank_fusion(candidate_lists, top_k=RRF_POOL)
        st.set(candidate_lists=[len(c) for c in candidate_lists], fused=len(fused))
        st.attach("chunks", "Fused candidate pool (RRF-ranked)", chunk_view(fused, n=25))

    # ── Cross-Encoder Reranking ──────────────────────────────────────────────
    with _Stage(emit, "rerank", "Cross-Encoder Reranking (bge-reranker-large)",
                "rerank_chunks", "fusion") as st:
        reranked = rerank_chunks(question, fused, top_k=FINAL_CONTEXT_CHUNKS)
        st.set(reranked=len(reranked))
        st.attach("chunks", "Final context (cross-encoder top chunks)", chunk_view(reranked, n=FINAL_CONTEXT_CHUNKS))

    # ── LLM Answer Generation ────────────────────────────────────────────────
    with _Stage(emit, "answer", "LLM Answer Generation", "generate_answer", "answer") as st:
        answer = generate_answer(analysis, reranked, strategy=need)
        st.set(confidence=answer.confidence_score, references=len(answer.references))
        st.attach("text", "Answer", answer.answer)
        st.attach("json", "References", [
            r.model_dump() if hasattr(r, "model_dump") else r for r in answer.references])

    elapsed = time.time() - t0
    logger.info("[pipeline] need=%s scope=%s domains=%s chunks=%d — %.1fs",
                need, prediction["scope"], graph_domains, len(reranked), elapsed)

    return {
        "answer": answer,
        "route": need,
        "domains": graph_domains,
        "analysis": analysis,
        "debug": {
            "retrieval_need": need,
            "query_form": need_choice.query_form,
            "domain_scope": prediction["scope"],
            "predicted_domains": prediction["domains"],
            "graph_domains": graph_domains,
            "domain_filter_disabled": disable_domain_filter,
            "graph_branch": GRAPH_BRANCH.get(need),
            "candidate_counts": [len(c) for c in candidate_lists],
            "fused": len(fused),
            "reranked": len(reranked),
            "graph_context": graph_context,
            "elapsed_s": round(elapsed, 2),
        },
    }


def _relations_to_context(relations: List[Dict[str, Any]]) -> str:
    lines = []
    for r in relations:
        lines.append(f"{r.get('subject','?')} --{r.get('label','related_to')}--> {r.get('object','?')}: "
                     f"{r.get('description','')}")
    return "\n".join(lines)
