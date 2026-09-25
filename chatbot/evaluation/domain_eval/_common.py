"""Shared helpers for the domain-filtering A/B evaluation scripts.

Loads domain_data/domain_eval_questions.json and runs every question through
src.pipeline.orchestrator.run_pipeline, either with domain filtering as
predicted (normal behavior) or forced off (disable_domain_filter=True).

Not meant to be run directly — see run_with_domain.py / run_without_domain.py /
run_comparison.py in this same directory.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CHATBOT_ROOT = Path(__file__).resolve().parents[2]  # .../chatbot
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from src.pipeline.orchestrator import run_pipeline  # noqa: E402

logger = logging.getLogger("agent_flow.domain_eval")

QUESTIONS_PATH = CHATBOT_ROOT / "domain_data" / "domain_eval_questions.json"
RESULTS_DIR = CHATBOT_ROOT / "domain_data" / "eval_results"


def load_questions(limit: Optional[int] = None) -> List[Dict[str, str]]:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = data["questions"]
    return questions[:limit] if limit else questions


def _run_one(item: Dict[str, str], disable_domain_filter: bool, tender_id: Optional[str]) -> Dict[str, Any]:
    t0 = time.time()
    try:
        result = run_pipeline(item["question"], tender_id=tender_id,
                              disable_domain_filter=disable_domain_filter)
        answer = result["answer"]
        debug = result["debug"]
        return {
            "id": item["id"],
            "expected_domain": item["domain"],
            "question": item["question"],
            "ok": True,
            "route": result.get("route"),
            "domain_scope": debug.get("domain_scope"),
            "predicted_domains": debug.get("predicted_domains"),
            "graph_domains": debug.get("graph_domains"),
            "domain_filter_disabled": debug.get("domain_filter_disabled"),
            "candidate_counts": debug.get("candidate_counts"),
            "fused": debug.get("fused"),
            "reranked": debug.get("reranked"),
            "confidence": answer.confidence_score,
            "answer": answer.answer,
            "references": [r.model_dump() if hasattr(r, "model_dump") else r for r in answer.references],
            "elapsed_s": round(time.time() - t0, 2),
        }
    except Exception as exc:
        logger.exception("pipeline failed for %s", item["id"])
        return {
            "id": item["id"],
            "expected_domain": item["domain"],
            "question": item["question"],
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.time() - t0, 2),
        }


def run_eval_set(questions: List[Dict[str, str]], disable_domain_filter: bool,
                 tender_id: Optional[str] = None, max_workers: int = 4) -> List[Dict[str, Any]]:
    """Run every eval question through the pipeline under ONE filtering condition."""
    label = "WITHOUT domain filtering" if disable_domain_filter else "WITH domain filtering"
    logger.info("Running %d questions %s (max_workers=%d)...", len(questions), label, max_workers)

    results: List[Optional[Dict[str, Any]]] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_one, item, disable_domain_filter, tender_id): i
            for i, item in enumerate(questions)
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            done += 1
            logger.info("[%d/%d] %s", done, len(questions), results[i]["id"])
    return results  # type: ignore[return-value]


def run_both_conditions(questions: List[Dict[str, str]], tender_id: Optional[str] = None,
                        max_workers: int = 4) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run every question under BOTH filtering conditions in one shared thread pool,
    so 'with' and 'without' tasks are genuinely interleaved rather than run back-to-back.

    Returns (with_domain_results, without_domain_results), each in question order.
    """
    id_to_index = {item["id"]: i for i, item in enumerate(questions)}
    tasks: List[Tuple[Dict[str, str], bool]] = (
        [(item, False) for item in questions] + [(item, True) for item in questions]
    )

    with_results: List[Optional[Dict[str, Any]]] = [None] * len(questions)
    without_results: List[Optional[Dict[str, Any]]] = [None] * len(questions)

    logger.info("Running %d questions x 2 conditions = %d pipeline calls (max_workers=%d)...",
               len(questions), len(tasks), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_one, item, disable, tender_id): (item, disable)
            for item, disable in tasks
        }
        done = 0
        for future in as_completed(futures):
            item, disable = futures[future]
            idx = id_to_index[item["id"]]
            result = future.result()
            (without_results if disable else with_results)[idx] = result
            done += 1
            cond = "without" if disable else "with"
            logger.info("[%d/%d] %s (%s)", done, len(tasks), item["id"], cond)

    return with_results, without_results  # type: ignore[return-value]


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d results -> %s", len(results), path)
