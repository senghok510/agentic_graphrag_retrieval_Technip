#!/usr/bin/env python3
"""Run ONE question through PPR (multi-hop graph) + Hybrid RAG, fused together --
the same combination orchestrator.py uses for the "multi_hop" retrieval-need
route. Mirrors that leg exactly: both branches run concurrently, their chunk
lists are RRF-fused into one pool, then cross-encoder reranked before answer
generation. No domain filtering (domains=None on the PPR call), matching the
"disable_domain_filter=True" default orchestrator.py itself uses for A/B eval.

Companion to run_hybrid_only_question.py -- run both scripts on the same
question to compare "Hybrid RAG alone" vs. "PPR + Hybrid RAG" answers
side by side.

    cd chatbot
    python scripts/domain_eval/run_ppr_hybrid_question.py
    python scripts/domain_eval/run_ppr_hybrid_question.py "some other question"
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

CHATBOT_ROOT = Path(__file__).resolve().parents[2]  # .../chatbot
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from src.pipeline.query_understanding import analyze_query  # noqa: E402
from src.pipeline.retrieval import hybrid_rag  # noqa: E402
from src.pipeline.ppr import run_multihop  # noqa: E402
from src.pipeline.fusion import reciprocal_rank_fusion, rerank_chunks  # noqa: E402
from src.pipeline.answer import generate_answer  # noqa: E402
from src.pipeline.clients import TARGET_ITB_ID  # noqa: E402

RRF_POOL = 50
FINAL_CONTEXT_CHUNKS = 20

DEFAULT_QUESTION = (
    "How are Change Orders related to the pricing of Re-measurable Works?"
)


def run_ppr_hybrid(question: str) -> dict:
    analysis = analyze_query(question)

    # Both branches run concurrently, same as orchestrator.py's multi_hop route.
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_hybrid = ex.submit(hybrid_rag, analysis, tender_id=TARGET_ITB_ID, top=RRF_POOL)
        f_ppr = ex.submit(run_multihop, analysis, domains=None, tender_id=TARGET_ITB_ID)
        hybrid_chunks = f_hybrid.result()
        ppr_result = f_ppr.result()

    ppr_chunks = ppr_result.get("chunks", [])
    ppr_relations = ppr_result.get("relations", [])

    # _chunks_for_relations (ppr.py) attaches "relation_ids" to each PPR chunk
    # row, but reciprocal_rank_fusion rebuilds chunk dicts into its own
    # normalised shape and drops that field -- capture the chunk->relation
    # mapping here, before fusion, so it can be reattached to the final
    # reranked chunks afterward. Hybrid-sourced chunks never have relation_ids
    # (they're not graph-derived), so this is PPR-provenance only, correctly.
    chunk_to_relation_ids: Dict[str, List[str]] = {
        c["chunk_id"]: c.get("relation_ids", []) for c in ppr_chunks if c.get("chunk_id")
    }
    relation_by_id: Dict[str, Dict[str, Any]] = {
        r["relation_id"]: r for r in ppr_relations if r.get("relation_id")
    }

    fused = reciprocal_rank_fusion([ppr_chunks, hybrid_chunks], top_k=RRF_POOL)
    reranked = rerank_chunks(question, fused, top_k=FINAL_CONTEXT_CHUNKS)

    for chunk in reranked:
        rel_ids = chunk_to_relation_ids.get(chunk.get("chunk_id"), [])
        chunk["asserted_relations"] = [relation_by_id[rid] for rid in rel_ids if rid in relation_by_id]

    answer = generate_answer(analysis, reranked, strategy="multi_hop")

    # run_multihop's "seeds" shape is inconsistent across its own return paths:
    # early returns give raw seed dicts (see ppr.py), the success path already
    # reduces them to plain entity-name strings. Handle both without assuming one.
    seeds_raw = ppr_result.get("seeds", [])
    seed_names = [s if isinstance(s, str) else s.get("entity_name") for s in seeds_raw]

    return {
        "question": question,
        "seeds": seed_names,
        "ppr_chunks": len(ppr_chunks),
        "hybrid_chunks": len(hybrid_chunks),
        "fused": len(fused),
        "reranked_chunks": reranked,
        "answer": answer,
    }


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    result = run_ppr_hybrid(question)
    answer = result["answer"]

    print("=" * 100)
    print("PPR (MULTI-HOP) + HYBRID RAG, FUSED")
    print("=" * 100)
    reranked_chunks = result["reranked_chunks"]
    print(f"Question: {result['question']}")
    print(f"PPR seed entities: {result['seeds']}")
    print(f"Chunks: ppr={result['ppr_chunks']} + hybrid={result['hybrid_chunks']} "
          f"-> fused={result['fused']} -> reranked={len(reranked_chunks)}")
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
    print()
    print(f"Final context ({len(reranked_chunks)} chunks fed to the LLM), with relations each chunk asserts:")
    print("-" * 100)
    for i, chunk in enumerate(reranked_chunks, 1):
        print(f"[{i}] {chunk.get('file_name')} (p.{chunk.get('page_number')})")
        print(f"    {(chunk.get('content') or '').strip()[:300]}")
        asserted = chunk.get("asserted_relations") or []
        if asserted:
            print(f"    Asserts {len(asserted)} relation(s):")
            for rel in asserted:
                print(f"      - {rel.get('subject')} --{rel.get('label')}--> {rel.get('object')}: "
                      f"{rel.get('description')}")
        print()
    print("=" * 100)


if __name__ == "__main__":
    main()
