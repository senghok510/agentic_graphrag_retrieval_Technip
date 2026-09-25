#!/usr/bin/env python3
"""Isolated test: does feeding keyword_scores into the domain-prediction prompt
improve accuracy against domain_eval_questions.json's expected_domain labels?

Does NOT touch src/pipeline/domain_routing.py, prompts.py, or run the full
retrieval pipeline -- it calls the domain-prediction LLM step directly, twice
per question (the real predict_graph_domains as baseline, vs. a keyword-signal
prompt variant defined only in this file), and scores both against the eval
set. Cheap and fast: no retrieval / rerank / answer generation / judge runs,
and no cross-encoder involved, so it's safe to parallelize (unlike
run_comparison.py, which must stay at --max-workers 1 because of the reranker).

    cd chatbot
    python scripts/domain_eval/experiments/test_domain_prediction.py [--limit N] [--max-workers 4]

Writes domain_data/eval_results/domain_prediction_experiment.json and prints a
side-by-side accuracy comparison + which questions flipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

CHATBOT_ROOT = Path(__file__).resolve().parents[3]  # .../chatbot
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from src.pipeline.clients import call_json, validate_model  # noqa: E402
from src.pipeline.domain_routing import (  # noqa: E402
    _domain_descriptions, _keyword_scores, graph_domain_nodes, predict_graph_domains,
)
from src.pipeline.schemas import GraphDomainPrediction  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent_flow.domain_eval.experiment")

QUESTIONS_PATH = CHATBOT_ROOT / "domain_data" / "domain_eval_questions.json"
RESULTS_PATH = CHATBOT_ROOT / "domain_data" / "eval_results" / "domain_prediction_experiment.json"

# ── Experimental prompt variant -- lives ONLY here, not in src/pipeline/prompts.py ──

_SYS_PROMPT = (
    "You predict which engineering domain(s) a GRAPH search should be scoped to for an "
    "EPC/ITB tender question. This scopes ONLY graph retrieval -- full-text search always "
    "covers everything. Return strict JSON only."
)

_USER_PROMPT_WITH_KEYWORDS = """Decide how to scope the GRAPH search for this question.

- scope = "single" : the question clearly concerns ONE engineering domain.
- scope = "multi"  : it spans a few related domains (list them).
- scope = "full"   : it is general, cross-cutting, or you are unsure -- search the entire graph.

Only use domain names from this list (drop anything not on it). When scope="full", return an empty domains list.

Some candidates below carry a "[keyword signal: N hit(s)]" annotation -- a count of
domain-specific terms found verbatim in the question. Treat it as supporting evidence,
not a rule: prefer the domain whose description fits AND has keyword support; a domain
with no keyword hits can still be right if its description clearly fits and no
better-evidenced domain does.

Candidate domains:
{domain_block}

Question: {question}

Return STRICT JSON:
{{"scope": "single|multi|full", "domains": ["<exact domain name>", ...], "reasoning": "one line"}}
"""


def predict_with_keyword_signal(question: str) -> Dict[str, Any]:
    """Local variant of predict_graph_domains() that annotates each candidate
    domain with its keyword-hit count. Mirrors the real function's post-
    processing exactly so the comparison isolates ONLY the prompt difference."""
    known = set(graph_domain_nodes())
    descriptions = _domain_descriptions()
    kw_scores = _keyword_scores(question)
    offer = [(n, descriptions.get(n, "")) for n in descriptions if not known or n in known]
    domain_block = "\n".join(
        f"- {name}: {desc}" + (f"  [keyword signal: {int(kw_scores[name])} hit(s)]" if kw_scores.get(name) else "")
        for name, desc in offer
    ) or "\n".join(f"- {n}" for n in known)

    try:
        data = call_json(_SYS_PROMPT, _USER_PROMPT_WITH_KEYWORDS.format(question=question, domain_block=domain_block))
        pred = validate_model(data, GraphDomainPrediction)
        scope, domains, reasoning = pred.scope, list(pred.domains), pred.reasoning
    except Exception as exc:
        logger.warning("predict_with_keyword_signal fallback (%s)", exc)
        scope, domains, reasoning = "full", [], f"fallback: {exc}"

    if known:
        domains = [d for d in domains if d in known]
    if scope == "full" or not domains:
        scope, domains = "full", []
    elif scope == "single" and len(domains) > 1:
        domains = domains[:1]

    return {"scope": scope, "domains": domains, "reasoning": reasoning, "keyword_scores": kw_scores}


def load_questions(limit: Optional[int] = None) -> List[Dict[str, str]]:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    qs = data["questions"]
    return qs[:limit] if limit else qs


def _score_one(item: Dict[str, str], variant_fn: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
    expected = item["domain"]
    try:
        pred = variant_fn(item["question"])
        predicted = pred.get("domains") or []
        scope = pred.get("scope")
        hit = None if scope == "full" else (expected in predicted)
        return {"id": item["id"], "expected": expected, "scope": scope,
                "predicted": predicted, "hit": hit, "reasoning": pred.get("reasoning", "")}
    except Exception as exc:
        return {"id": item["id"], "expected": expected, "error": f"{type(exc).__name__}: {exc}"}


def run_variant(questions: List[Dict[str, str]], variant_fn: Callable[[str], Dict[str, Any]],
                label: str, max_workers: int) -> List[Dict[str, Any]]:
    logger.info("Running variant '%s' over %d questions (max_workers=%d)...", label, len(questions), max_workers)
    results: List[Optional[Dict[str, Any]]] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_score_one, item, variant_fn): i for i, item in enumerate(questions)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            done += 1
            logger.info("[%s %d/%d] %s", label, done, len(questions), results[i]["id"])
    return results  # type: ignore[return-value]


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = sum(1 for r in results if r.get("hit") is True)
    wrong = sum(1 for r in results if r.get("hit") is False)
    full = sum(1 for r in results if r.get("hit") is None and "error" not in r)
    errors = sum(1 for r in results if "error" in r)
    n_scoped = correct + wrong
    return {
        "scoped_correct": correct,
        "scoped_wrong": wrong,
        "full_scope": full,
        "errors": errors,
        "accuracy_when_scoped": round(correct / n_scoped, 4) if n_scoped else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only test the first N questions")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="safe to parallelize -- this test never touches the cross-encoder reranker")
    args = parser.parse_args()

    questions = load_questions(limit=args.limit)

    baseline_results = run_variant(
        questions, lambda q: predict_graph_domains(q), "baseline (current prompt)", args.max_workers)
    variant_results = run_variant(
        questions, predict_with_keyword_signal, "keyword-signal variant", args.max_workers)

    baseline_summary = summarize(baseline_results)
    variant_summary = summarize(variant_results)

    by_id_base = {r["id"]: r for r in baseline_results}
    by_id_var = {r["id"]: r for r in variant_results}
    flipped_to_correct, flipped_to_wrong = [], []
    for qid in by_id_base:
        b, v = by_id_base[qid], by_id_var.get(qid, {})
        if b.get("hit") is False and v.get("hit") is True:
            flipped_to_correct.append(qid)
        elif b.get("hit") is True and v.get("hit") is False:
            flipped_to_wrong.append(qid)

    report = {
        "n_questions": len(questions),
        "baseline": baseline_summary,
        "keyword_signal_variant": variant_summary,
        "flipped_wrong_to_correct": flipped_to_correct,
        "flipped_correct_to_wrong": flipped_to_wrong,
        "baseline_results": baseline_results,
        "variant_results": variant_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 72)
    print(f"Domain prediction experiment -- {len(questions)} questions")
    print("=" * 72)
    print(f"{'':<28}{'baseline':>12}{'keyword-signal':>18}")
    for key in ("scoped_correct", "scoped_wrong", "full_scope", "errors", "accuracy_when_scoped"):
        print(f"{key:<28}{str(baseline_summary[key]):>12}{str(variant_summary[key]):>18}")
    print()
    print(f"Flipped WRONG -> correct ({len(flipped_to_correct)}): {flipped_to_correct}")
    print(f"Flipped correct -> WRONG ({len(flipped_to_wrong)}): {flipped_to_wrong}")
    print()
    print(f"Wrote full detail -> {RESULTS_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
