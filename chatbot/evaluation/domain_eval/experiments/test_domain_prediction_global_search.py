#!/usr/bin/env python3
"""Isolated test: infer a query's domain from the domains of the relations
retrieval ALREADY trusts, instead of comparing the query to domain
descriptions (RRF + verifier, test_domain_prediction_rrf_verifier.py).

Mechanism:
  1. analyze_query(question)  -> high_level_keywords (same production call
     the real pipeline already makes for the graph branch -- reused as-is,
     not reimplemented, since fidelity to the real query-understanding stage
     matters here).
  2. _global_search(hl_keywords, top_k, domains=None) -> top-K relations,
     UNSCOPED (domains=None disables the CLASSIFIED_AS filter entirely --
     this unscoped call IS the domain-inference step, distinct from the
     real, domain-scoped retrieval that would follow it in production).
  3. Look up each of those relations' existing CLASSIFIED_AS domain(s).
  4. Majority vote across the top-K -> predicted domain(s).

Why this might beat RRF+verifier: RRF+verifier grounds the decision in a
domain DESCRIPTION file, which can drift out of sync with what's actually
classified in the graph (the whole "phantom domains" / candidate-bounding
mismatch this session kept finding). This grounds the decision in the SAME
relation embeddings and SAME CLASSIFIED_AS edges the real graph retrieval
uses, so it's automatically calibrated to current graph state -- at the cost
of an extra (unscoped) retrieval pass before the real one, and a small-k
majority vote that could be noisy on ambiguous/multi-topic questions.

No LLM verifier call here (unlike the RRF+verifier experiment) -- the vote
is a plain count, no extra classification step -- so this is also cheaper.

    cd chatbot
    python scripts/domain_eval/experiments/test_domain_prediction_global_search.py \\
        [--limit N] [--max-workers 4] [--top-k 5]

Writes domain_data/eval_results/domain_prediction_experiment_global_search.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

CHATBOT_ROOT = Path(__file__).resolve().parents[3]  # .../chatbot
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from src.pipeline.clients import neo4j_driver  # noqa: E402
from src.pipeline.graph_retrieval import _global_search  # noqa: E402
from src.pipeline.query_understanding import analyze_query  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent_flow.domain_eval.experiment_global_search")

DOMAIN_DATA_DIR = CHATBOT_ROOT / "domain_data"
QUESTIONS_PATH = DOMAIN_DATA_DIR / "domain_eval_questions.json"
RESULTS_PATH = DOMAIN_DATA_DIR / "eval_results" / "domain_prediction_experiment_global_search.json"

_RELATION_DOMAINS_QUERY = """
UNWIND $ids AS rid
MATCH (r:Relation {relationId: rid})-[:CLASSIFIED_AS]->(dom)
RETURN rid, collect(DISTINCT dom.domainName) AS domains
"""


def relation_domains(relation_ids: List[str]) -> Dict[str, List[str]]:
    """Existing CLASSIFIED_AS domain(s) for each relation -- the signal the
    majority vote is built from. A relation can be classified into a domain
    at either tier (LowLevelDomain or MidLevelDomain), matching how the eval
    questions themselves aren't tier-restricted."""
    if not relation_ids:
        return {}
    with neo4j_driver().session() as s:
        rows = s.run(_RELATION_DOMAINS_QUERY, {"ids": relation_ids}).data()
    return {row["rid"]: row["domains"] for row in rows}


def predict_global_search(question: str, top_k: int = 5) -> Dict[str, Any]:
    analysis = analyze_query(question)
    hl_keywords = analysis.get("high_level_keywords") or []

    if not hl_keywords:
        return {
            "scope": "general", "domains": [], "reasoning": "no high_level_keywords from analyze_query",
            "top_relations": [], "domain_votes": {},
        }

    top_relations = _global_search(hl_keywords, top_k=top_k, domains=None)
    if not top_relations:
        return {
            "scope": "general", "domains": [], "reasoning": "global search returned no relations",
            "top_relations": [], "domain_votes": {},
        }

    rel_ids = [r["relation_id"] for r in top_relations]
    rel_domains = relation_domains(rel_ids)

    votes: Counter = Counter()
    per_relation = []
    for rel in top_relations:
        doms = rel_domains.get(rel["relation_id"], [])
        votes.update(doms)
        per_relation.append({
            "relation_id": rel["relation_id"],
            "subject": rel.get("subject"), "object": rel.get("object"), "label": rel.get("label"),
            "vector_score": rel.get("vector_score"),
            "domains": doms,
        })

    if not votes:
        return {
            "scope": "general", "domains": [], "reasoning": "no top-k relation had a CLASSIFIED_AS domain",
            "top_relations": per_relation, "domain_votes": {},
        }

    top_domain, top_count = votes.most_common(1)[0]
    reasoning = f"{top_count}/{top_k} top relations classified as {top_domain!r}"
    return {
        "scope": "single", "domains": [top_domain], "reasoning": reasoning,
        "top_relations": per_relation, "domain_votes": dict(votes),
        "high_level_keywords": hl_keywords,
    }


def load_questions(limit: Optional[int] = None) -> List[Dict[str, str]]:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    qs = data["questions"]
    return qs[:limit] if limit else qs


def _score_one(item: Dict[str, str], top_k: int) -> Dict[str, Any]:
    expected = item["domain"]
    try:
        pred = predict_global_search(item["question"], top_k=top_k)
        predicted = pred.get("domains") or []
        scope = pred.get("scope")
        all_topk_domains = {d for r in pred.get("top_relations", []) for d in r.get("domains", [])}

        # Same convention as test_domain_prediction_rrf_verifier.py: GENERAL is
        # correctly "scope=general" (no clear winner); every other expected
        # domain must actually win the vote to count as a hit.
        if expected == "GENERAL":
            hit = (scope == "general") or ("GENERAL" in predicted)
        else:
            hit = None if scope == "general" else (expected in predicted)

        return {
            "id": item["id"], "expected": expected, "scope": scope, "predicted": predicted, "hit": hit,
            "domain_votes": pred.get("domain_votes"), "reasoning": pred.get("reasoning", ""),
            "candidate_recall": expected in all_topk_domains,
            "top_relations": pred.get("top_relations"),
        }
    except Exception as exc:
        return {"id": item["id"], "expected": expected, "error": f"{type(exc).__name__}: {exc}"}


def run_variant(questions: List[Dict[str, str]], top_k: int, max_workers: int) -> List[Dict[str, Any]]:
    logger.info("Running global-search domain inference over %d questions (top_k=%d, max_workers=%d)...",
                len(questions), top_k, max_workers)
    results: List[Optional[Dict[str, Any]]] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_score_one, item, top_k): i for i, item in enumerate(questions)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            done += 1
            logger.info("[global-search %d/%d] %s", done, len(questions), results[i]["id"])
    return results  # type: ignore[return-value]


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = sum(1 for r in results if r.get("hit") is True)
    wrong = sum(1 for r in results if r.get("hit") is False)
    general = sum(1 for r in results if r.get("hit") is None and "error" not in r)
    errors = sum(1 for r in results if "error" in r)
    n_scoped = correct + wrong
    summary = {
        "scoped_correct": correct,
        "scoped_wrong": wrong,
        "general_scope": general,
        "errors": errors,
        "accuracy_when_scoped": round(correct / n_scoped, 4) if n_scoped else None,
    }
    recall_rows = [r for r in results if "candidate_recall" in r]
    if recall_rows:
        n_recalled = sum(1 for r in recall_rows if r["candidate_recall"])
        summary["candidate_recall_at_k"] = round(n_recalled / len(recall_rows), 4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="only test the first N questions")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="safe to parallelize -- Azure OpenAI/Neo4j clients are thread-safe")
    parser.add_argument("--top-k", type=int, default=5, help="top-K relations from unscoped global search")
    args = parser.parse_args()

    questions = load_questions(limit=args.limit)

    results = run_variant(questions, top_k=args.top_k, max_workers=args.max_workers)

    summary = summarize(results)
    missed_by_recall = [r["id"] for r in results if r.get("candidate_recall") is False]

    report = {
        "n_questions": len(questions),
        "top_k": args.top_k,
        "summary": summary,
        "missed_because_not_in_top_k": missed_by_recall,
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 72)
    print(f"Domain prediction experiment (unscoped global-search top-{args.top_k} majority vote) "
          f"-- {len(questions)} questions")
    print("=" * 72)
    for key in ("scoped_correct", "scoped_wrong", "general_scope", "errors", "accuracy_when_scoped",
               "candidate_recall_at_k"):
        print(f"{key:<28}{str(summary.get(key))}")
    print()
    print(f"Expected domain never present in top-{args.top_k} relations ({len(missed_by_recall)}): {missed_by_recall}")
    print()
    print(f"Wrote full detail -> {RESULTS_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
