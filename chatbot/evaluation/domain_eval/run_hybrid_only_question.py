#!/usr/bin/env python3
"""Run ONE question through the Hybrid RAG branch only -- Query Understanding ->
hybrid_rag (BM25 + dense vector, full Azure index) -> RRF -> rerank -> answer.
No graph retrieval (LightRAG/PPR) at all, and no domain filtering (Hybrid RAG
is never domain-scoped in the real pipeline either -- see orchestrator.py).

Mirrors orchestrator.py's Hybrid RAG leg exactly (same functions, same
RRF_POOL/FINAL_CONTEXT_CHUNKS constants, same classify_retrieval_need ->
generate_answer strategy wiring) but skips the graph branch and the fan-out
over both, so this is what "Hybrid RAG alone" actually means for the
qualitative with/without-graph comparison.

    cd chatbot
    python scripts/domain_eval/run_hybrid_only_question.py
    python scripts/domain_eval/run_hybrid_only_question.py "some other question"
"""

from __future__ import annotations

import sys
from pathlib import Path

CHATBOT_ROOT = Path(__file__).resolve().parents[2]  # .../chatbot
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from src.pipeline.query_understanding import analyze_query, classify_retrieval_need  # noqa: E402
from src.pipeline.retrieval import hybrid_rag  # noqa: E402
from src.pipeline.fusion import reciprocal_rank_fusion, rerank_chunks  # noqa: E402
from src.pipeline.answer import generate_answer  # noqa: E402
from src.pipeline.clients import TARGET_ITB_ID  # noqa: E402

RRF_POOL = 50
FINAL_CONTEXT_CHUNKS = 20

DEFAULT_QUESTION = (
    "How are Change Orders related to the pricing of Re-measurable Works?"
)


def run_hybrid_only(question: str) -> dict:
    analysis = analyze_query(question)
    need = classify_retrieval_need(question).need

    hybrid_chunks = hybrid_rag(analysis, tender_id=TARGET_ITB_ID, top=RRF_POOL)

    fused = reciprocal_rank_fusion([hybrid_chunks], top_k=RRF_POOL)
    reranked = rerank_chunks(question, fused, top_k=FINAL_CONTEXT_CHUNKS)

    answer = generate_answer(analysis, reranked, strategy=need)

    return {
        "question": question,
        "retrieval_need": need,
        "hybrid_chunks": len(hybrid_chunks),
        "fused": len(fused),
        "reranked_chunks": reranked,
        "answer": answer,
    }


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    result = run_hybrid_only(question)
    answer = result["answer"]

    print("=" * 100)
    print("HYBRID RAG ONLY")
    print("=" * 100)
    print(f"Question: {result['question']}")
    print(f"Retrieval need (routing, informational only -- not used to pick a branch here): {result['retrieval_need']}")
    print(f"Chunks: hybrid={result['hybrid_chunks']} -> fused={result['fused']} -> reranked={len(result['reranked_chunks'])}")
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
