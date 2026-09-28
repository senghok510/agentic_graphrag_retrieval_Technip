"""LangGraph orchestration for the adaptive tender retrieval pipeline.

Retrieval and answer implementations remain ordinary Python functions.
LangGraph owns their execution order, parallel fan-out/fan-in, and shared state.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .answer import generate_answer
from .clients import TARGET_ITB_ID
from .domain_routing import predict_graph_domains
from .fusion import reciprocal_rank_fusion, rerank_chunks
from .lightrag import run_aggregation, run_single_hop
from .ppr import run_multihop
from .query_understanding import analyze_query, classify_retrieval_need
from .retrieval import hybrid_rag
from .trace import EmitFn
from .trace import Stage as _Stage
from .trace import noop as _noop
from .views import chunk_view, entity_view, relation_view

logger = logging.getLogger("agent_flow.pipeline.langgraph")

RRF_POOL = 50
FINAL_CONTEXT_CHUNKS = 20
HYBRID_TOP = 15

GRAPH_BRANCH = {
    "single_hop": "Local Graph Retrieval",
    "aggregation": "Dual-Level Retrieval",
    "multi_hop": "Hub-aware PPR",
}


class PipelineState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes for one pipeline run."""

    question: str
    tender_id: str
    top: int
    disable_domain_filter: bool
    emit: EmitFn
    analysis: dict[str, Any]
    need_choice: Any
    need: str
    prediction: dict[str, Any]
    graph_domains: list[str] | None
    hybrid_chunks: list[dict[str, Any]]
    graph_result: dict[str, Any]
    candidate_lists: list[list[dict[str, Any]]]
    graph_context: str
    fused: list[dict[str, Any]]
    reranked: list[dict[str, Any]]
    answer: Any
    debug: dict[str, Any]


def _emit(state: PipelineState) -> EmitFn:
    return state.get("emit") or _noop


def _relations_to_context(relations: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{r.get('subject', '?')} --{r.get('label', 'related_to')}--> "
        f"{r.get('object', '?')}: {r.get('description', '')}"
        for r in relations
    )


def _run_graph_branch(
    need: str,
    analysis: dict,
    domains: list[str] | None,
    tender_id: str,
    emit: EmitFn,
) -> dict[str, Any]:
    if need == "single_hop":
        return run_single_hop(analysis, domains=domains, emit=emit)
    if need == "aggregation":
        return run_aggregation(analysis, domains=domains, emit=emit)
    if need == "multi_hop":
        return run_multihop(analysis, domains=domains, emit=emit, tender_id=tender_id)
    return {"chunks": [], "relations": [], "context": ""}


def understand_node(state: PipelineState) -> dict[str, Any]:
    with _Stage(
        _emit(state),
        "query_understanding",
        "Query Understanding",
        "analyze_query",
        "understanding",
    ) as stage:
        analysis = analyze_query(state["question"])
        stage.set(
            is_complex=analysis.get("is_complex"),
            entity_hints=analysis.get("entity_hints", []),
            high_level_keywords=analysis.get("high_level_keywords", []),
            sub_questions=analysis.get("sub_questions", []),
        )
        stage.attach(
            "json",
            "Query analysis",
            {
                "entity_hints": analysis.get("entity_hints", []),
                "category_hints": analysis.get("category_hints", []),
                "high_level_keywords": analysis.get("high_level_keywords", []),
                "sub_questions": analysis.get("sub_questions", []),
                "expanded_query": analysis.get("expanded_query", ""),
            },
        )
        stage.attach(
            "text",
            "HyDE document (used for the vector query)",
            analysis.get("hyde_doc", ""),
        )
    return {"analysis": analysis}


def classify_need_node(state: PipelineState) -> dict[str, Any]:
    with _Stage(
        _emit(state),
        "retrieval_need",
        "Retrieval Need Classification",
        "classify_retrieval_need",
        "classification",
    ) as stage:
        choice = classify_retrieval_need(state["question"])
        stage.set(
            need=choice.need,
            query_form=choice.query_form,
            reasoning=choice.reasoning,
        )
    return {"need_choice": choice, "need": choice.need}


def predict_domains_node(state: PipelineState) -> dict[str, Any]:
    with _Stage(
        _emit(state),
        "domain_prediction",
        "Graph Domain Prediction",
        "predict_graph_domains",
        "classification",
    ) as stage:
        prediction = predict_graph_domains(state["question"], state["analysis"])
        stage.set(
            scope=prediction["scope"],
            domains=prediction["domains"],
            reasoning=prediction.get("reasoning", ""),
        )
        stage.attach(
            "json",
            "Domain prediction",
            {
                "scope": prediction["scope"],
                "domains": prediction["domains"],
                "keyword_scores": prediction.get("keyword_scores", {}),
                "reasoning": prediction.get("reasoning", ""),
            },
        )
    return {"prediction": prediction}


def select_scope_node(state: PipelineState) -> dict[str, Any]:
    prediction = state["prediction"]
    graph_domains = None if prediction["scope"] == "full" else (prediction["domains"] or None)
    if state["disable_domain_filter"]:
        graph_domains = None

    with _Stage(
        _emit(state),
        "graph_scope",
        "Graph Scope Selection",
        "graph_scope",
        "classification",
    ) as stage:
        stage.set(
            scope=prediction["scope"],
            domains=graph_domains or "full graph",
            graph_branch=GRAPH_BRANCH.get(state["need"], "— (no graph, Hybrid only)"),
            filter_disabled=state["disable_domain_filter"],
        )
    return {"graph_domains": graph_domains}


def hybrid_retrieval_node(state: PipelineState) -> dict[str, Any]:
    with _Stage(
        _emit(state),
        "hybrid_rag",
        "Hybrid RAG (BM25 + Dense, full index)",
        "hybrid_rag",
        "retrieval",
    ) as stage:
        rows = hybrid_rag(state["analysis"], tender_id=state["tender_id"], top=state["top"])
        stage.set(chunks=len(rows), scope="full Azure index")
        stage.attach("chunks", "Hybrid RAG chunks (full index)", chunk_view(rows))
    return {"hybrid_chunks": rows}


def graph_retrieval_node(state: PipelineState) -> dict[str, Any]:
    need = state["need"]
    if need == "textual_factoid":
        return {"graph_result": {"chunks": [], "relations": [], "context": ""}}

    with _Stage(
        _emit(state),
        "graph_retrieval",
        f"Graph Retrieval — {GRAPH_BRANCH[need]}",
        "_run_graph_branch",
        "graph",
    ) as stage:
        result = _run_graph_branch(
            need,
            state["analysis"],
            state.get("graph_domains"),
            state["tender_id"],
            _emit(state),
        )
        stage.set(
            chunks=len(result.get("chunks", [])),
            domains=state.get("graph_domains") or "full",
        )
        relations = result.get("relations") or (
            result.get("local_rels", []) + result.get("global_rels", [])
        )
        stage.attach(
            "entities",
            "Linked / seed entities",
            entity_view(result.get("linked_entities") or result.get("seeds") or []),
        )
        stage.attach("relations", "Graph relations", relation_view(relations))
        stage.attach(
            "chunks",
            "Graph evidence chunks",
            chunk_view(result.get("chunks", [])),
        )
    return {"graph_result": result}


def fusion_node(state: PipelineState) -> dict[str, Any]:
    graph_result = state.get("graph_result", {})
    candidate_lists = (
        [state["hybrid_chunks"]]
        if state["need"] == "textual_factoid"
        else [graph_result.get("chunks", []), state["hybrid_chunks"]]
    )
    with _Stage(
        _emit(state),
        "evidence_fusion",
        "Evidence Fusion & Deduplication (RRF)",
        "reciprocal_rank_fusion",
        "fusion",
    ) as stage:
        fused = reciprocal_rank_fusion(candidate_lists, top_k=RRF_POOL)
        stage.set(
            candidate_lists=[len(candidates) for candidates in candidate_lists],
            fused=len(fused),
        )
        stage.attach("chunks", "Fused candidate pool (RRF-ranked)", chunk_view(fused, n=25))
    graph_context = graph_result.get("context") or _relations_to_context(
        graph_result.get("relations", [])
    )
    return {
        "candidate_lists": candidate_lists,
        "graph_context": graph_context,
        "fused": fused,
    }


def rerank_node(state: PipelineState) -> dict[str, Any]:
    with _Stage(
        _emit(state),
        "rerank",
        "Cross-Encoder Reranking (bge-reranker-large)",
        "rerank_chunks",
        "fusion",
    ) as stage:
        reranked = rerank_chunks(state["question"], state["fused"], top_k=FINAL_CONTEXT_CHUNKS)
        stage.set(reranked=len(reranked))
        stage.attach(
            "chunks",
            "Final context (cross-encoder top chunks)",
            chunk_view(reranked, n=FINAL_CONTEXT_CHUNKS),
        )
    return {"reranked": reranked}


def answer_node(state: PipelineState) -> dict[str, Any]:
    with _Stage(
        _emit(state),
        "answer",
        "LLM Answer Generation",
        "generate_answer",
        "answer",
    ) as stage:
        answer = generate_answer(state["analysis"], state["reranked"], strategy=state["need"])
        stage.set(
            confidence=answer.confidence_score,
            references=len(answer.references),
        )
        stage.attach("text", "Answer", answer.answer)
        stage.attach(
            "json",
            "References",
            [
                reference.model_dump() if hasattr(reference, "model_dump") else reference
                for reference in answer.references
            ],
        )

    prediction = state["prediction"]
    candidate_lists = state["candidate_lists"]
    return {
        "answer": answer,
        "debug": {
            "retrieval_need": state["need"],
            "query_form": state["need_choice"].query_form,
            "domain_scope": prediction["scope"],
            "predicted_domains": prediction["domains"],
            "graph_domains": state.get("graph_domains"),
            "domain_filter_disabled": state["disable_domain_filter"],
            "graph_branch": GRAPH_BRANCH.get(state["need"]),
            "candidate_counts": [len(candidates) for candidates in candidate_lists],
            "fused": len(state["fused"]),
            "reranked": len(state["reranked"]),
            "graph_context": state.get("graph_context", ""),
        },
    }


def build_pipeline_graph():
    """Build the graph with static joins for predictable synchronization."""

    builder = StateGraph(PipelineState)
    builder.add_node("understand", understand_node)
    builder.add_node("classify_need", classify_need_node)
    builder.add_node("predict_domains", predict_domains_node)
    builder.add_node("select_scope", select_scope_node)
    builder.add_node("hybrid_retrieval", hybrid_retrieval_node)
    builder.add_node("graph_retrieval", graph_retrieval_node)
    builder.add_node("fusion", fusion_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "understand")
    builder.add_edge("understand", "classify_need")
    builder.add_edge("understand", "predict_domains")
    builder.add_edge(["classify_need", "predict_domains"], "select_scope")
    builder.add_edge("select_scope", "hybrid_retrieval")
    builder.add_edge("select_scope", "graph_retrieval")
    builder.add_edge(["hybrid_retrieval", "graph_retrieval"], "fusion")
    builder.add_edge("fusion", "rerank")
    builder.add_edge("rerank", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


pipeline_graph = build_pipeline_graph()


def run_langgraph_pipeline(
    question: str,
    tender_id: str | None = None,
    top: int = HYBRID_TOP,
    emit: EmitFn | None = None,
    disable_domain_filter: bool = True,
) -> dict[str, Any]:
    """Run the compiled graph and return the legacy orchestrator result shape."""

    started_at = time.time()
    result = pipeline_graph.invoke(
        {
            "question": question,
            "tender_id": tender_id or TARGET_ITB_ID,
            "top": top,
            "disable_domain_filter": disable_domain_filter,
            "emit": emit or _noop,
        }
    )
    elapsed = time.time() - started_at
    debug = dict(result["debug"])
    debug["elapsed_s"] = round(elapsed, 2)
    logger.info(
        "[pipeline] need=%s scope=%s domains=%s chunks=%d — %.1fs",
        result["need"],
        result["prediction"]["scope"],
        result.get("graph_domains"),
        len(result["reranked"]),
        elapsed,
    )
    return {
        "answer": result["answer"],
        "route": result["need"],
        "domains": result.get("graph_domains"),
        "analysis": result["analysis"],
        "debug": debug,
    }
