#!/usr/bin/env python3
"""Isolated test: two-stage domain prediction, fully standalone -- no functions
from src/pipeline/domain_routing.py or any other production module, and no
Neo4j dependency at all. Reads domain_description.json / domain_keywords.json
directly, so every described domain is a legitimate candidate (production's
graph_domain_nodes() silently drops any domain that isn't yet a real graph
node -- 11 of 40 currently, per scripts/domain_eval/experiments/check_domain_graph_nodes.py
-- and this experiment deliberately does not inherit that gap).

NOTE: an earlier version of this file had a Stage 0 "general vs specific" gate that
ran a blind LLM call before any embedding/keyword work, to short-circuit obviously
general questions. Removed -- it was net-negative: it has no visibility into the
actual domain list, so it fires on surface language ("broad", "spans multiple
areas") even when a specific domain genuinely covers the question. Concretely, it
flipped RISK_MANAGEMENT-02 from correct to wrong (gated it as general purely
because the question mentioned "various project risks", never letting RRF/the
verifier see that RISK MANAGEMENT is a real, well-matched domain) while only
fixing one other case. Stage 2's own "general" fallback is strictly better at this
judgment because it only ever fires after seeing real ranked candidates.

Stage 1 (retrieval): rank all candidate domains two ways --
  * semantic similarity  (question embedding vs. each domain description's embedding)
  * keyword similarity   (lexical keyword hit-count, hyphen/space-normalized)
then fuse the two rankings with Reciprocal Rank Fusion (same technique already
used for chunk fusion in src/pipeline/fusion.py) and take the top-K candidates.

Stage 2 (verification): show ONLY those top-K candidates (with descriptions) to
an LLM verifier, which picks exactly one domain ("single") or says none of them
fit after all ("general") -- a much easier, more accurate decision than scanning
all ~40 domains at once, since two independent signals already agreed on a short,
high-quality shortlist.

Scores against domain_eval_questions.json's expected_domain labels. There is no
"baseline" comparison in this file by design (see above) -- run
test_domain_prediction.py if you want a baseline-vs-variant comparison against
the production prompt. Also reports "candidate recall" -- how often the
expected domain even made the top-K shortlist -- since that's the ceiling the
verifier stage can't exceed.

    cd chatbot
    python scripts/domain_eval/experiments/test_domain_prediction_rrf_verifier.py \\
        [--limit N] [--max-workers 4] [--top-k 5] [--rrf-k 10]

Writes domain_data/eval_results/domain_prediction_experiment_rrf_verifier.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel

CHATBOT_ROOT = Path(__file__).resolve().parents[3]  # .../chatbot
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

# The only reuse from src/ is generic Azure OpenAI plumbing (HTTP client wrappers) --
# not domain-prediction logic, and not anything that touches Neo4j.
from src.pipeline.clients import call_json, embed_texts_large_model, validate_model, _l2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent_flow.domain_eval.experiment")

DOMAIN_DATA_DIR = CHATBOT_ROOT / "domain_data"
QUESTIONS_PATH = DOMAIN_DATA_DIR / "domain_eval_questions.json"
RESULTS_PATH = CHATBOT_ROOT / "domain_data" / "eval_results" / "domain_prediction_experiment_rrf_verifier.json"


class SingleDomainVerification(BaseModel):
    """Local schema for this experiment's binary single/general verifier --
    intentionally NOT src.pipeline.schemas.GraphDomainPrediction, whose scope is
    Literal["single","multi","full"] and wouldn't accept "general" (or a plain
    single `domain` string instead of a `domains` list)."""

    scope: Literal["single", "general"]
    domain: Optional[str] = None
    reasoning: str = ""


# ── Domain data, loaded directly from the JSON files (no Neo4j, no domain_routing.py) ──

@lru_cache(maxsize=1)
def _domain_descriptions() -> Dict[str, str]:
    return json.loads((DOMAIN_DATA_DIR / "domain_description.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _domain_keywords() -> Dict[str, List[str]]:
    return json.loads((DOMAIN_DATA_DIR / "domain_keywords.json").read_text(encoding="utf-8"))


def _keyword_scores(question: str) -> Dict[str, float]:
    """Lexical keyword hit-count per domain. Hyphenated keywords (e.g. "access-control")
    also match the naturally-written, space-separated form real questions use."""
    q = re.sub(r"[^a-z0-9\s\-]", " ", question.lower())
    q_tokens = set(re.split(r"\s+", q))
    q_spaced = q.replace("-", " ")
    scores: Dict[str, float] = {}
    for domain, keywords in _domain_keywords().items():
        hits = 0
        for kw in keywords:
            kw_l = kw.lower()
            if " " in kw_l or "-" in kw_l:
                if kw_l in q or kw_l.replace("-", " ") in q_spaced:
                    hits += 1
            elif kw_l in q_tokens:
                hits += 1
        if hits:
            scores[domain] = float(hits)
    return scores


# ── Stage 1a: semantic ranking (embeddings, cached once per process) ─────────

_DOMAIN_EMB: Dict[str, np.ndarray] = {}
_DOMAIN_EMB_LOCK = threading.Lock()


def _domain_embeddings() -> Dict[str, np.ndarray]:
    """Embed every domain's 'name: description' once, thread-safe, cached process-wide."""
    global _DOMAIN_EMB
    if not _DOMAIN_EMB:
        with _DOMAIN_EMB_LOCK:
            if not _DOMAIN_EMB:
                descriptions = _domain_descriptions()
                names = list(descriptions.keys())
                texts = [f"{name}: {descriptions[name]}" for name in names]
                logger.info("Embedding %d domain descriptions (one-time)...", len(names))
                vectors = embed_texts_large_model(texts)
                _DOMAIN_EMB = {name: _l2(np.array(vec)) for name, vec in zip(names, vectors)}
                logger.info("Domain embeddings cached.")
    return _DOMAIN_EMB


def _semantic_rank(question: str) -> List[str]:
    domain_embs = _domain_embeddings()
    q_vec = _l2(np.array(embed_texts_large_model([question])[0]))
    scored = [(name, float(np.dot(q_vec, vec))) for name, vec in domain_embs.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored]


# ── Stage 1b: keyword ranking ─────────────────────────────────────────────────

def _keyword_rank(question: str) -> List[str]:
    kw_scores = _keyword_scores(question)
    items = sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in items]


# ── Stage 1c: RRF fusion (same formula as fusion.reciprocal_rank_fusion) ─────

def _rrf_fuse(ranked_lists: List[List[str]], k: int = 10) -> List[str]:
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, name in enumerate(ranked, start=1):
            scores[name] = scores.get(name, 0.0) + 1.0 / (k + rank)
    return [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ── Stage 2: LLM verifier over the top-K shortlist only ──────────────────────

_VERIFIER_SYS = (
    "You choose the best-fitting engineering domain for an EPC/ITB tender question, "
    "from a short pre-filtered candidate list. Return strict JSON only."
)

_VERIFIER_USER = """A retrieval step pre-selected these candidate domains as the most plausible
matches for the question below (semantic + keyword relevance, most likely first). Decide:

- scope = "single"  : exactly one candidate clearly fits -- name it in "domain".
- scope = "general" : NONE of these candidates actually fit well -- search the entire graph
  instead. Set "domain" to null.

Candidates (ranked, most likely first):
{domain_block}

Question: {question}

Return STRICT JSON:
{{"scope": "single|general", "domain": "<exact single domain name from the candidates above, or null>", "reasoning": "one line"}}
"""


def predict_rrf_verifier(question: str, top_k: int = 5, rrf_k: int = 10) -> Dict[str, Any]:
    descriptions = _domain_descriptions()

    semantic_ranked = _semantic_rank(question)
    keyword_ranked = _keyword_rank(question)
    fused = _rrf_fuse([semantic_ranked, keyword_ranked], k=rrf_k)
    top_candidates = fused[:top_k]

    base = {
        "candidates": top_candidates,
        "semantic_ranked": semantic_ranked[:top_k],
        "keyword_ranked": keyword_ranked[:top_k],
    }

    if not top_candidates:
        return {**base, "scope": "general", "domains": [], "reasoning": "no candidates from RRF fusion"}

    domain_block = "\n".join(f"- {name}: {descriptions.get(name, '')}" for name in top_candidates)

    try:
        data = call_json(_VERIFIER_SYS, _VERIFIER_USER.format(domain_block=domain_block, question=question))
        pred = validate_model(data, SingleDomainVerification)
        verifier_scope, domain, reasoning = pred.scope, pred.domain, pred.reasoning
    except Exception as exc:
        logger.warning("predict_rrf_verifier fallback (%s)", exc)
        verifier_scope, domain, reasoning = "general", None, f"fallback: {exc}"

    if verifier_scope == "single" and domain in top_candidates:
        scope, domains = "single", [domain]
    else:
        scope, domains = "general", []

    return {**base, "scope": scope, "domains": domains, "reasoning": reasoning}


def load_questions(limit: Optional[int] = None) -> List[Dict[str, str]]:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    qs = data["questions"]
    return qs[:limit] if limit else qs


def _score_rrf(item: Dict[str, str], top_k: int, rrf_k: int) -> Dict[str, Any]:
    expected = item["domain"]
    try:
        pred = predict_rrf_verifier(item["question"], top_k=top_k, rrf_k=rrf_k)
        predicted = pred.get("domains") or []
        scope = pred.get("scope")
        candidates = pred.get("candidates") or []
        # GENERAL is a special case: "scope=general" (no single domain fits, search
        # everything) IS the semantically correct answer for a question whose expected
        # label is GENERAL -- that's what the GENERAL domain concept means. For every
        # other expected domain, scope=general stays unscored (falling back isn't
        # "correct" just because it avoided a wrong pick).
        if expected == "GENERAL":
            hit = (scope == "general") or ("GENERAL" in predicted)
        else:
            hit = None if scope == "general" else (expected in predicted)
        return {
            "id": item["id"], "expected": expected, "scope": scope, "predicted": predicted, "hit": hit,
            "candidates": candidates, "candidate_recall": expected in candidates,
            "semantic_ranked": pred.get("semantic_ranked"), "keyword_ranked": pred.get("keyword_ranked"),
            "reasoning": pred.get("reasoning", ""),
        }
    except Exception as exc:
        return {"id": item["id"], "expected": expected, "error": f"{type(exc).__name__}: {exc}"}


def run_variant(questions: List[Dict[str, str]], score_fn: Callable[[Dict[str, str]], Dict[str, Any]],
                label: str, max_workers: int) -> List[Dict[str, Any]]:
    logger.info("Running '%s' over %d questions (max_workers=%d)...", label, len(questions), max_workers)
    results: List[Optional[Dict[str, Any]]] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(score_fn, item): i for i, item in enumerate(questions)}
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only test the first N questions")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="safe to parallelize -- no cross-encoder reranker involved")
    parser.add_argument("--top-k", type=int, default=5, help="candidates handed to the LLM verifier")
    parser.add_argument("--rrf-k", type=int, default=10, help="RRF k constant (smaller = sharper top ranks)")
    args = parser.parse_args()

    questions = load_questions(limit=args.limit)

    # Pre-warm the domain embedding cache once, sequentially, before any threading --
    # avoids every worker racing to embed the same 40 descriptions on first use.
    _domain_embeddings()

    results = run_variant(
        questions, lambda item: _score_rrf(item, args.top_k, args.rrf_k),
        f"RRF (semantic+keyword) top-{args.top_k} -> verifier", args.max_workers)

    summary = summarize(results)
    missed_by_recall = [r["id"] for r in results if r.get("candidate_recall") is False]

    report = {
        "n_questions": len(questions),
        "top_k": args.top_k,
        "rrf_k": args.rrf_k,
        "summary": summary,
        "missed_because_not_in_top_k": missed_by_recall,
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 72)
    print(f"Domain prediction experiment (RRF + verifier, standalone) -- {len(questions)} questions, top_k={args.top_k}")
    print("=" * 72)
    for key in ("scoped_correct", "scoped_wrong", "general_scope", "errors", "accuracy_when_scoped",
               "candidate_recall_at_k"):
        print(f"{key:<28}{str(summary.get(key))}")
    print()
    print(f"Expected domain never made the top-{args.top_k} shortlist ({len(missed_by_recall)}): {missed_by_recall}")
    print()
    print(f"Wrote full detail -> {RESULTS_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
