#!/usr/bin/env python3
"""Run ONE question through Dual-Level Retrieval (LightRAG local+global,
run_aggregation) + Hybrid RAG, fused together -- the same combination
orchestrator.py uses for the "aggregation" retrieval-need route. Mirrors
run_ppr_hybrid_question.py's structure exactly, swapping run_multihop for
run_aggregation.

Companion to run_hybrid_only_question.py -- run both on the same question to
compare "Hybrid RAG alone" vs. "Dual-Level + Hybrid RAG" answers side by side.

    cd chatbot
    python scripts/domain_eval/run_dual_level_hybrid_question.py
    python scripts/domain_eval/run_dual_level_hybrid_question.py "some other question"
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHATBOT_ROOT = Path(__file__).resolve().parents[2]  # .../chatbot
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from src.pipeline.query_understanding import analyze_query  # noqa: E402
from src.pipeline.retrieval import hybrid_rag  # noqa: E402
from src.pipeline.lightrag import run_aggregation  # noqa: E402
from src.pipeline.fusion import reciprocal_rank_fusion, rerank_chunks  # noqa: E402
from src.pipeline.answer import generate_answer  # noqa: E402
from src.pipeline.clients import TARGET_ITB_ID  # noqa: E402

RRF_POOL = 50
FINAL_CONTEXT_CHUNKS = 25

DEFAULT_QUESTION = (
    "What are the height-based requirements and spill control considerations "
    "for the Air-Cooled Condenser (ACC) in the tender context?"
)


def run_dual_level_hybrid(question: str) -> dict:
    analysis = analyze_query(question)

    # Both branches run concurrently, same as orchestrator.py's aggregation route.
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_hybrid = ex.submit(hybrid_rag, analysis, tender_id=TARGET_ITB_ID, top=RRF_POOL)
        f_dual = ex.submit(run_aggregation, analysis, domains=None)
        hybrid_chunks = f_hybrid.result()
        dual_result = f_dual.result()

    dual_chunks = dual_result.get("chunks", [])

    fused = reciprocal_rank_fusion([dual_chunks, hybrid_chunks], top_k=RRF_POOL)
    # reranked = rerank_chunks(question, fused, top_k=FINAL_CONTEXT_CHUNKS)

    answer = generate_answer(analysis, fused, strategy="aggregation")

    return {
        "question": question,
        "linked_entities": dual_result.get("linked_entities", []),
        "local_rels": len(dual_result.get("local_rels", [])),
        "global_rels": len(dual_result.get("global_rels", [])),
        "dual_chunks": len(dual_chunks),
        "hybrid_chunks": len(hybrid_chunks),
        "fused": len(fused),
        "reranked_chunks": fused,
        "answer": answer,
    }


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    result = run_dual_level_hybrid(question)
    answer = result["answer"]

    print("=" * 100)
    print("DUAL-LEVEL RETRIEVAL (LightRAG local+global) + HYBRID RAG, FUSED")
    print("=" * 100)
    print(f"Question: {result['question']}")
    print(f"Linked entities: {result['linked_entities']}")
    print(f"Relations: local={result['local_rels']} global={result['global_rels']}")
    print(f"Chunks: dual={result['dual_chunks']} + hybrid={result['hybrid_chunks']} "
          f"-> fused={result['fused']} -> reranked={len(result['reranked_chunks'])}")
    print()
    print(f"Confidence: {answer.confidence_score}")
    print()
    print("Answer:")
    print(answer.answer)
    print()
    print(f"References ({len(answer.references)}):")
    for ref in answer.references:
        ref_dict = ref.model_dump() if hasattr(ref, "model_dump") else ref
        print(f"  - {ref_dict}")
    print("=" * 100)


if __name__ == "__main__":
    main()
