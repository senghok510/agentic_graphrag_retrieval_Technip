#!/usr/bin/env python3
"""Isolated test: RRF -> verifier restricted to single/multi (no "general" escape
hatch), run only on the domain-specific subset of the eval set (excludes the 2
GENERAL-labeled questions).

Simulates "the user has already confirmed this question is about a specific
domain" -- there is no real UI for that yet, so this test approximates it by
construction: every remaining question SHOULD map to exactly one real domain,
so the verifier is never given a bail-out option. This isolates the actual
question at stake for the "ask the user first" design: once general-vs-specific
is no longer the verifier's job, how accurately does it commit to single vs
multi and pick the right domain(s)?

Stage 1 (retrieval) is UNCHANGED from test_domain_prediction_rrf_verifier.py --
semantic ranking + keyword ranking, fused with RRF, top-K candidates. Only
Stage 2 (verification) differs: scope is single|multi (never general), and a
"multi" answer can name up to MAX_MULTI_DOMAINS candidates.

Uses domain_description_ROC_INPEX.json / domain_keywords_ROC_INPEX.json (the
pruned, Neo4j-scoped files) -- the same files src/pipeline/domain_routing.py
now points at -- so this test reflects what production would actually see.

Scoring collapses to one number (no more "general_scope" bucket): a question is
correct if its expected_domain is anywhere in the predicted domain list,
whether that list has one item (single) or several (multi).

    cd chatbot
    python scripts/domain_eval/experiments/test_domain_prediction_single_multi.py \\
        [--limit N] [--max-workers 4] [--top-k 10] [--rrf-k 10]

Writes domain_data/eval_results/domain_prediction_experiment_single_multi.json
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
RESULTS_PATH = CHATBOT_ROOT / "domain_data" / "eval_results" / "domain_prediction_experiment_single_multi.json"

MAX_MULTI_DOMAINS = 2


class SingleMultiVerification(BaseModel):
    """Local schema for this experiment's forced single/multi verifier -- no
    "general" option at all, since the premise is that general-vs-specific has
    already been decided (by the user, upstream of this stage)."""

    scope: Literal["single", "multi"]
    domains: List[str]
    reasoning: str = ""


# ── Domain data, loaded directly from the pruned ROC_INPEX JSON files ────────

@lru_cache(maxsize=1)
def _domain_descriptions() -> Dict[str, str]:
    return json.loads((DOMAIN_DATA_DIR / "domain_description_ROC_INPEX.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _domain_keywords() -> Dict[str, List[str]]:
    return json.loads((DOMAIN_DATA_DIR / "domain_keywords_ROC_INPEX.json").read_text(encoding="utf-8"))


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


# ── Stage 2: LLM verifier, forced single/multi -- no general escape hatch ────

_VERIFIER_SYS = (
    "You choose the best-fitting engineering domain(s) for an EPC/ITB tender question, "
    "from a short pre-filtered candidate list. The question is already known to be about "
    "one of these domains -- you must not decline to answer. Return strict JSON only."
)

_VERIFIER_USER = """A retrieval step pre-selected these candidate domains as the most plausible
matches for the question below (semantic + keyword relevance, most likely first). The user has
already confirmed this question concerns a specific engineering/technical/commercial domain, so
you must pick from the candidates -- do not say none of them fit.

- scope = "single" : exactly one candidate clearly fits.
- scope = "multi"   : the question genuinely spans a few of these candidates (max {max_multi}).

Candidates (ranked, most likely first):
{domain_block}

Question: {question}

Return STRICT JSON:
{{"scope": "single|multi", "domains": ["<exact domain name from the candidates above>", ...], "reasoning": "one line"}}
"""


def predict_single_multi(question: str, top_k: int = 10, rrf_k: int = 10) -> Dict[str, Any]:
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
        # Should not happen with a non-empty domain list, but stay defensive.
        return {**base, "scope": "single", "domains": [], "reasoning": "no candidates from RRF fusion"}

    domain_block = "\n".join(f"- {name}: {descriptions.get(name, '')}" for name in top_candidates)

    try:
        data = call_json(_VERIFIER_SYS, _VERIFIER_USER.format(
            domain_block=domain_block, question=question, max_multi=MAX_MULTI_DOMAINS))
        pred = validate_model(data, SingleMultiVerification)
        scope, domains, reasoning = pred.scope, list(pred.domains), pred.reasoning
    except Exception as exc:
        logger.warning("predict_single_multi fallback (%s) -> top RRF candidate", exc)
        # No "general" bail-out available -- fall back to the top RRF candidate
        # rather than returning nothing, since the premise is this question DOES
        # have a domain.
        scope, domains, reasoning = "single", top_candidates[:1], f"fallback: {exc}"

    # Only accept domains the verifier was actually offered; enforce the multi cap.
    domains = [d for d in domains if d in top_candidates]
    if not domains:
        scope, domains = "single", top_candidates[:1]
    elif scope == "single" and len(domains) > 1:
        domains = domains[:1]
    elif scope == "multi":
        domains = domains[:MAX_MULTI_DOMAINS]

    return {**base, "scope": scope, "domains": domains, "reasoning": reasoning}


def load_questions(limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Only the domain-specific subset -- excludes GENERAL, since this verifier
    has no way to say "none of these fit"."""
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    qs = [q for q in data["questions"] if q["domain"] != "GENERAL"]
    return qs[:limit] if limit else qs


def _score(item: Dict[str, str], top_k: int, rrf_k: int) -> Dict[str, Any]:
    expected = item["domain"]
    try:
        pred = predict_single_multi(item["question"], top_k=top_k, rrf_k=rrf_k)
        predicted = pred.get("domains") or []
        scope = pred.get("scope")
        candidates = pred.get("candidates") or []
        hit = expected in predicted
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
    errors = sum(1 for r in results if "error" in r)
    n_scored = correct + wrong
    summary: Dict[str, Any] = {
        "correct": correct,
        "wrong": wrong,
        "errors": errors,
        "accuracy": round(correct / n_scored, 4) if n_scored else None,
    }
    for scope_name in ("single", "multi"):
        rows = [r for r in results if r.get("scope") == scope_name and "error" not in r]
        summary[f"n_{scope_name}"] = len(rows)
        if rows:
            c = sum(1 for r in rows if r["hit"])
            summary[f"{scope_name}_accuracy"] = round(c / len(rows), 4)
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
    parser.add_argument("--top-k", type=int, default=10, help="candidates handed to the LLM verifier")
    parser.add_argument("--rrf-k", type=int, default=10, help="RRF k constant (smaller = sharper top ranks)")
    args = parser.parse_args()

    questions = load_questions(limit=args.limit)
    logger.info("Loaded %d domain-specific questions (GENERAL-labeled ones excluded)", len(questions))

    # Pre-warm the domain embedding cache once, sequentially, before any threading --
    # avoids every worker racing to embed the same domain descriptions on first use.
    _domain_embeddings()

    results = run_variant(
        questions, lambda item: _score(item, args.top_k, args.rrf_k),
        f"RRF top-{args.top_k} -> single/multi verifier", args.max_workers)

    summary = summarize(results)
    missed_by_recall = [r["id"] for r in results if r.get("candidate_recall") is False]

    report = {
        "n_questions": len(questions),
        "top_k": args.top_k,
        "rrf_k": args.rrf_k,
        "max_multi_domains": MAX_MULTI_DOMAINS,
        "summary": summary,
        "missed_because_not_in_top_k": missed_by_recall,
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 72)
    print(f"Domain prediction experiment (single/multi, no general) -- {len(questions)} questions, top_k={args.top_k}")
    print("=" * 72)
    for key in ("correct", "wrong", "errors", "accuracy", "n_single", "single_accuracy",
               "n_multi", "multi_accuracy", "candidate_recall_at_k"):
        print(f"{key:<20}{str(summary.get(key))}")
    print()
    print(f"Expected domain never made the top-{args.top_k} shortlist ({len(missed_by_recall)}): {missed_by_recall}")

    wrong_rows = [r for r in results if r.get("hit") is False]
    if wrong_rows:
        print()
        print(f"Wrong predictions ({len(wrong_rows)}):")
        for r in sorted(wrong_rows, key=lambda r: r["id"]):
            print(f"  [{r['id']}] expected={r['expected']!r} scope={r['scope']} predicted={r['predicted']}")

    print()
    print(f"Wrote full detail -> {RESULTS_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
