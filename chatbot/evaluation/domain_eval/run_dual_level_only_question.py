#!/usr/bin/env python3
"""Run ONE question through Dual-Level Retrieval (LightRAG local+global,
run_aggregation) ALONE -- no Hybrid RAG contribution at all. Companion to
run_dual_level_hybrid_question.py (which fuses Dual-Level with Hybrid RAG,
matching production's "aggregation" route) -- this one isolates what
Dual-Level Retrieval contributes entirely on its own, for comparison against
the Hybrid-RAG-only baseline (run_hybrid_only_question.py).

Single-source RRF (no Hybrid RAG to fuse against) -- kept for consistency
with the RRF-then-rerank convention the rest of the pipeline uses, same
pattern as run_hybrid_only_question.py's single-source RRF over just its one
hybrid_rag list.

    cd chatbot
    python scripts/domain_eval/run_dual_level_only_question.py
    python scripts/domain_eval/run_dual_level_only_question.py "some other question"
"""

from __future__ import annotations

import sys
from pathlib import Path

CHATBOT_ROOT = Path(__file__).resolve().parents[2]  # .../chatbot
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from src.pipeline.query_understanding import analyze_query  # noqa: E402
from src.pipeline.lightrag import run_aggregation  # noqa: E402
from src.pipeline.fusion import reciprocal_rank_fusion, rerank_chunks  # noqa: E402
from src.pipeline.answer import generate_answer  # noqa: E402

RRF_POOL = 50
FINAL_CONTEXT_CHUNKS = 30

DEFAULT_QUESTION = (
    "What are the height-based requirements and spill control considerations "
    "for the Air-Cooled Condenser (ACC) in the tender context?"
)


def run_dual_level_only(question: str) -> dict:
    analysis = analyze_query(question)

    dual_result = run_aggregation(analysis, domains=None)
    dual_chunks = dual_result.get("chunks", [])

    fused = reciprocal_rank_fusion([dual_chunks], top_k=RRF_POOL)
    # reranked = rerank_chunks(question, fused, top_k=FINAL_CONTEXT_CHUNKS)

    answer = generate_answer(analysis, fused, strategy="aggregation")

    return {
        "question": question,
        "linked_entities": dual_result.get("linked_entities", []),
        "local_rels": len(dual_result.get("local_rels", [])),
        "global_rels": len(dual_result.get("global_rels", [])),
        "dual_chunks": len(dual_chunks),
        "fused": len(fused),
        "reranked_chunks": fused,
        "answer": answer,
    }


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    result = run_dual_level_only(question)
    answer = result["answer"]

    print("=" * 100)
    print("DUAL-LEVEL RETRIEVAL ONLY (LightRAG local+global, no Hybrid RAG)")
    print("=" * 100)
    print(f"Question: {result['question']}")
    print(f"Linked entities: {result['linked_entities']}")
    print(f"Relations: local={result['local_rels']} global={result['global_rels']}")
    print(f"Chunks: dual={result['dual_chunks']} -> fused={result['fused']} "
          f"-> reranked={len(result['reranked_chunks'])}")
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
