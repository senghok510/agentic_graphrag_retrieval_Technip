from __future__ import annotations

import json
import threading

import pytest
from src.models.schemas import ResponseGeneration
from src.pipeline import langgraph_pipeline as graph_module
from src.pipeline.schemas import RetrievalNeed


def _chunk(chunk_id: str, source: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "content": f"content for {chunk_id}",
        "file_name": "tender.pdf",
        "page_number": "1",
        "score": 1.0,
        "source": source,
    }


def _analysis(question: str) -> dict:
    return {
        "question": question,
        "expanded_query": question,
        "hyde_doc": question,
        "is_complex": False,
        "entity_hints": [],
        "category_hints": [],
        "high_level_keywords": [],
        "sub_questions": [],
    }


@pytest.fixture
def pipeline_stubs(monkeypatch):
    calls: list[tuple] = []
    route = {"value": "textual_factoid"}

    monkeypatch.setattr(graph_module, "analyze_query", _analysis)
    monkeypatch.setattr(
        graph_module,
        "classify_retrieval_need",
        lambda _question: RetrievalNeed(need=route["value"], query_form="none", reasoning="test"),
    )
    monkeypatch.setattr(
        graph_module,
        "predict_graph_domains",
        lambda _question, _analysis: {
            "scope": "single",
            "domains": ["PROCESS"],
            "reasoning": "test",
            "keyword_scores": {"PROCESS": 1.0},
        },
    )

    def hybrid(_analysis, tender_id=None, top=15):
        calls.append(("hybrid", tender_id, top))
        return [_chunk("hybrid-1", "hybrid_rag")]

    def single(_analysis, domains=None, emit=None):
        calls.append(("single_hop", domains))
        return {"chunks": [_chunk("graph-1", "local_graph")], "relations": []}

    def aggregation(_analysis, domains=None, emit=None):
        calls.append(("aggregation", domains))
        return {"chunks": [_chunk("graph-1", "dual_level")], "relations": []}

    def multihop(_analysis, domains=None, emit=None, tender_id=None):
        calls.append(("multi_hop", domains, tender_id))
        return {"chunks": [_chunk("graph-1", "ppr")], "relations": []}

    monkeypatch.setattr(graph_module, "hybrid_rag", hybrid)
    monkeypatch.setattr(graph_module, "run_single_hop", single)
    monkeypatch.setattr(graph_module, "run_aggregation", aggregation)
    monkeypatch.setattr(graph_module, "run_multihop", multihop)
    monkeypatch.setattr(
        graph_module,
        "rerank_chunks",
        lambda _question, chunks, top_k=20: chunks[:top_k],
    )
    monkeypatch.setattr(
        graph_module,
        "generate_answer",
        lambda _analysis, _chunks, strategy: ResponseGeneration(
            answer=f"answer via {strategy}", references=[], confidence_score=0.9
        ),
    )
    return route, calls


@pytest.mark.parametrize(
    ("route_name", "expected_branch", "candidate_counts"),
    [
        ("textual_factoid", None, [1]),
        ("single_hop", "single_hop", [1, 1]),
        ("aggregation", "aggregation", [1, 1]),
        ("multi_hop", "multi_hop", [1, 1]),
    ],
)
def test_all_routes_keep_legacy_result_and_trace_contract(
    pipeline_stubs, route_name, expected_branch, candidate_counts
):
    route, calls = pipeline_stubs
    route["value"] = route_name
    events = []

    result = graph_module.run_langgraph_pipeline(
        "What does the tender require?",
        tender_id="TENDER-1",
        top=7,
        emit=events.append,
        disable_domain_filter=False,
    )

    assert result["route"] == route_name
    assert result["domains"] == ["PROCESS"]
    assert result["answer"].answer == f"answer via {route_name}"
    assert result["debug"]["candidate_counts"] == candidate_counts
    assert isinstance(result["debug"]["elapsed_s"], float)
    assert ("hybrid", "TENDER-1", 7) in calls

    branch_calls = [call[0] for call in calls if call[0] != "hybrid"]
    assert branch_calls == ([] if expected_branch is None else [expected_branch])

    stage_events = [event for event in events if event["type"] == "stage"]
    for stage_name in (
        "query_understanding",
        "retrieval_need",
        "domain_prediction",
        "graph_scope",
        "hybrid_rag",
        "evidence_fusion",
        "rerank",
        "answer",
    ):
        statuses = {event["status"] for event in stage_events if event["stage"] == stage_name}
        assert statuses == {"start", "done"}

    graph_statuses = {
        event["status"] for event in stage_events if event["stage"] == "graph_retrieval"
    }
    assert graph_statuses == (set() if route_name == "textual_factoid" else {"start", "done"})


def test_domain_filter_can_still_be_disabled(pipeline_stubs):
    route, calls = pipeline_stubs
    route["value"] = "single_hop"

    result = graph_module.run_langgraph_pipeline("question", disable_domain_filter=True)

    assert result["domains"] is None
    assert ("single_hop", None) in calls
    assert result["debug"]["domain_filter_disabled"] is True


def test_retrieval_branches_execute_in_parallel(pipeline_stubs, monkeypatch):
    route, _calls = pipeline_stubs
    route["value"] = "single_hop"
    barrier = threading.Barrier(2)

    def hybrid(_analysis, tender_id=None, top=15):
        barrier.wait(timeout=2)
        return [_chunk("hybrid-1", "hybrid_rag")]

    def single(_analysis, domains=None, emit=None):
        barrier.wait(timeout=2)
        return {"chunks": [_chunk("graph-1", "local_graph")], "relations": []}

    monkeypatch.setattr(graph_module, "hybrid_rag", hybrid)
    monkeypatch.setattr(graph_module, "run_single_hop", single)

    result = graph_module.run_langgraph_pipeline("question")

    assert result["debug"]["candidate_counts"] == [1, 1]


def test_graph_structure_contains_parallel_joins():
    mermaid = graph_module.pipeline_graph.get_graph().draw_mermaid()

    assert "understand --> classify_need" in mermaid
    assert "understand --> predict_domains" in mermaid
    assert "hybrid_retrieval --> fusion" in mermaid
    assert "graph_retrieval --> fusion" in mermaid


@pytest.mark.asyncio
async def test_sse_adapter_keeps_existing_event_contract(monkeypatch):
    from src.pipeline import streaming

    monkeypatch.setattr(
        streaming,
        "run_pipeline",
        lambda question, tender_id=None, emit=None: {
            "answer": ResponseGeneration(
                answer="final answer", references=[], confidence_score=0.8
            ),
            "route": "textual_factoid",
            "debug": {
                "domain_scope": "full",
                "graph_domains": None,
                "elapsed_s": 0.1,
            },
        },
    )

    frames = [frame async for frame in streaming.stream_pipeline_sse("question", "TENDER-1")]
    payloads = [
        json.loads(frame.removeprefix("data: ").strip())
        for frame in frames
        if "[DONE]" not in frame
    ]

    assert payloads[0] == {
        "type": "open",
        "question": "question",
        "tender_id": "TENDER-1",
    }
    assert payloads[-1]["type"] == "final"
    assert payloads[-1]["answer"] == "final answer"
    assert payloads[-1]["route"] == "textual_factoid"
