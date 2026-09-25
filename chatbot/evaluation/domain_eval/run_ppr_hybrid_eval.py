#!/usr/bin/env python3
"""Batch-evaluate run_ppr_hybrid_question.py's PPR + Hybrid RAG pipeline over
every question in data/eval_multihop_100_ppr.jsonl, scored by _llm_judge.py
(completeness/relevance/fluency/accuracy, pointwise -- no gold-answer
comparison, per that module's own docstring; the dataset's reference_answer
is carried through into the report for eyeballing but not used by the judge).

Sequential over questions, not parallel -- rerank_chunks' cross-encoder isn't
thread-safe on this machine and segfaults under concurrency, the same
constraint as every other pipeline eval script in scripts/domain_eval/.
Judging afterward IS parallelized (plain LLM calls, no reranker involved).

Records average judge scores and average latency (elapsed_s) across all
questions, plus full per-question detail, to a JSON file.

    cd chatbot
    python scripts/domain_eval/run_ppr_hybrid_eval.py
    python scripts/domain_eval/run_ppr_hybrid_eval.py --limit 5   # smoke test first
    python scripts/domain_eval/run_ppr_hybrid_eval.py --judge-max-workers 8

Writes domain_data/eval_results/ppr_hybrid_multihop_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CHATBOT_ROOT = Path(__file__).resolve().parents[2]  # .../chatbot
REPO_ROOT = CHATBOT_ROOT.parent
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from run_ppr_hybrid_question import run_ppr_hybrid  # noqa: E402
import _llm_judge  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent_flow.domain_eval.ppr_hybrid_eval")

QUESTIONS_PATH = REPO_ROOT / "data" / "eval_multihop_100_ppr.jsonl"
RESULTS_DIR = CHATBOT_ROOT / "domain_data" / "eval_results"
RESULTS_PATH = RESULTS_DIR / "ppr_hybrid_multihop_eval.json"


def load_questions(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def _run_one(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]
    t0 = time.time()
    try:
        result = run_ppr_hybrid(question)
        answer = result["answer"]
        return {
            "id": idx,
            "question": question,
            "reference_answer": item.get("reference_answer"),
            "question_type": item.get("question_type"),
            "n_hops": item.get("n_hops"),
            "ok": True,
            "seeds": result["seeds"],
            "ppr_chunks": result["ppr_chunks"],
            "hybrid_chunks": result["hybrid_chunks"],
            "fused": result["fused"],
            "reranked": len(result["reranked_chunks"]),
            "confidence": answer.confidence_score,
            "answer": answer.answer,
            "references": [r.model_dump() if hasattr(r, "model_dump") else r for r in answer.references],
            "elapsed_s": round(time.time() - t0, 2),
        }
    except Exception as exc:
        logger.exception("run_ppr_hybrid failed for question %d", idx)
        return {
            "id": idx,
            "question": question,
            "reference_answer": item.get("reference_answer"),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.time() - t0, 2),
        }


def run_all(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sequential -- see module docstring re: reranker thread-safety."""
    results = []
    for i, item in enumerate(questions, 1):
        logger.info("[%d/%d] %s", i, len(questions), item["question"][:80])
        results.append(_run_one(i, item))
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_results = [r for r in results if r.get("ok")]
    judged = [r for r in ok_results if r.get("judge")]

    summary: Dict[str, Any] = {
        "n_questions": len(results),
        "n_ok": len(ok_results),
        "n_errors": len(results) - len(ok_results),
        "n_judged": len(judged),
    }

    if ok_results:
        summary["avg_latency_s"] = round(sum(r["elapsed_s"] for r in ok_results) / len(ok_results), 2)

    if judged:
        for dim in _llm_judge.ALL_SCORE_KEYS:
            scores = [r["judge"][dim] for r in judged if r["judge"].get(dim) is not None]
            if scores:
                summary[f"avg_{dim}"] = round(sum(scores) / len(scores), 3)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--judge-max-workers", type=int, default=4,
                        help="judge is a plain LLM call, no reranker involved -- safe to parallelize")
    args = parser.parse_args()

    questions = load_questions(limit=args.limit)
    logger.info("Loaded %d questions from %s", len(questions), QUESTIONS_PATH)

    results = run_all(questions)

    _llm_judge.judge_all(results, max_workers=args.judge_max_workers)

    summary = summarize(results)

    report = {
        "questions_path": str(QUESTIONS_PATH),
        "n_questions": len(questions),
        "summary": summary,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print()
    print("=" * 100)
    print("PPR + HYBRID RAG -- multihop eval summary")
    print("=" * 100)
    for k, v in summary.items():
        print(f"{k:<20}{v}")
    print()
    print(f"Wrote full detail -> {RESULTS_PATH}")
    print("=" * 100)


if __name__ == "__main__":
    main()
