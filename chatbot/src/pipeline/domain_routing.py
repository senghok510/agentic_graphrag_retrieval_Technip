"""Stage: single-domain gate + Domain Prediction + Domain-aware Hybrid Search.

Mermaid nodes:
  * ``Can this query be answered using Hybrid Semantic Search within a single
    engineering domain?`` → ``can_answer_in_single_domain``
  * ``Domain Prediction (Engineering Domain)``                → ``predict_domain``
  * ``Domain-aware Hybrid Search (BM25 + Vector Search)``     → ``domain_aware_hybrid_search``

Domain prediction is NOT present in ppr.py / evaluate_lightrag.py, so it is built
here from ``domain_data/domain_keywords.json`` and ``domain_data/domain_description.json``
(lexical keyword vote + LLM disambiguation over the domain descriptions).
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .clients import call_json, validate_model
from .retrieval import hybrid_search, _normalise_search_rows
from .prompts import DOMAIN_PREDICTION_SYSTEM_PROMPT, DOMAIN_PREDICTION_USER_PROMPT
from .schemas import GraphDomainPrediction

logger = logging.getLogger("agent_flow.pipeline.domain_routing")

_DOMAIN_DIR = Path(__file__).resolve().parents[2] / "domain_data"


@lru_cache(maxsize=1)
def _domain_keywords() -> Dict[str, List[str]]:
    try:
        return json.loads((_DOMAIN_DIR / "domain_keywords_ROC_INPEX.json").read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        logger.warning("could not load domain_keywords.json: %s", exc)
        return {}


@lru_cache(maxsize=1)
def _domain_descriptions() -> Dict[str, str]:
    try:
        return json.loads((_DOMAIN_DIR / "domain_description_ROC_INPEX.json").read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        logger.warning("could not load domain_description.json: %s", exc)
        return {}


def domain_names() -> List[str]:
    return list(_domain_descriptions().keys()) or list(_domain_keywords().keys())


def _keyword_scores(question: str) -> Dict[str, float]:
    q = re.sub(r"[^a-z0-9\s\-]", " ", question.lower())
    q_tokens = set(re.split(r"\s+", q))
    # Hyphenated keywords (e.g. "access-control") must also match the naturally-written,
    # space-separated form ("access control") that real questions actually use.
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


_DOMAIN_PREDICT_SYS = (
    "You classify tendering/EPC questions into engineering domains. "
    "Return strict JSON only."
)

_DOMAIN_PREDICT_USER = """Given the question and the candidate engineering domains (with descriptions),
decide which SINGLE domain the question primarily belongs to, and whether it can be answered
within that one domain using plain semantic search (no cross-domain relational reasoning).

Question: {question}

Domains:
{domain_block}

Return STRICT JSON:
{{"domain": "<exact domain name or 'MULTIPLE'>", "single_domain": true|false, "confidence": 0.0, "reasoning": "one line"}}
"""


def predict_domain(question: str) -> Dict[str, Any]:
    """Predict the engineering domain for a question.

    Combines a lexical keyword vote with an LLM decision over the domain
    descriptions. Returns ``{domain, single_domain, confidence, reasoning,
    keyword_scores}``.
    """
    kw_scores = _keyword_scores(question)
    descriptions = _domain_descriptions()
    domain_block = "\n".join(f"- {name}: {desc}" for name, desc in descriptions.items())

    try:
        data = call_json(
            system_prompt=_DOMAIN_PREDICT_SYS,
            user_prompt=_DOMAIN_PREDICT_USER.format(question=question, domain_block=domain_block),
        )
    except Exception as exc:
        logger.warning("predict_domain LLM fallback (%s)", exc)
        data = {}

    domain = data.get("domain") or (max(kw_scores, key=kw_scores.get) if kw_scores else "MULTIPLE")
    single_domain = bool(data.get("single_domain", False))
    confidence = float(data.get("confidence", 0.0) or 0.0)

    # A domain the LLM invented but that we don't know about → treat as MULTIPLE.
    if domain not in descriptions and domain != "MULTIPLE":
        domain = max(kw_scores, key=kw_scores.get) if kw_scores else "MULTIPLE"

    return {
        "domain": domain,
        "single_domain": single_domain and domain != "MULTIPLE",
        "confidence": confidence,
        "reasoning": data.get("reasoning", ""),
        "keyword_scores": kw_scores,
    }


def can_answer_in_single_domain(analysis: dict, prediction: Optional[dict] = None) -> Tuple[bool, dict]:
    """(v1, deprecated) The old single-domain gate. Superseded in v2 by
    ``predict_graph_domains`` — domain prediction no longer gates the whole
    pipeline, it only scopes graph retrieval."""
    prediction = prediction or predict_domain(analysis.get("question", ""))
    yes = (
        not analysis.get("is_complex", False)
        and prediction.get("single_domain", False)
        and prediction.get("confidence", 0.0) >= 0.5
    )
    return yes, prediction


def domain_aware_hybrid_search(analysis: dict, prediction: dict,
                               tender_id: Optional[str] = None, top: int = 20) -> List[Dict[str, Any]]:
    """(v1, deprecated) Domain-biased hybrid search.

    NOT used in v2 — per the spec, Hybrid RAG must always search the full Azure
    index and never be restricted by domain. Kept only for backward reference.
    """
    domain = prediction.get("domain", "")
    domain_terms = _domain_keywords().get(domain, [])[:8]
    base_query = analysis.get("expanded_query") or analysis.get("question", "")
    domain_query = f"{base_query} {' '.join(domain_terms)}".strip()
    rows = hybrid_search(query=domain_query, tender_id=tender_id, top=top,
                         vector_query_text=analysis.get("hyde_doc"))
    normalised = _normalise_search_rows(rows, source="domain_hybrid")
    for r in normalised:
        r["domain"] = domain
    return normalised


# ── v2: Graph Domain Prediction (scopes graph retrieval only) ────────────────

@lru_cache(maxsize=1)
def graph_domain_nodes() -> tuple:
    """Distinct domain names present as (Low|Mid|High)LevelDomain nodes in Neo4j.

    These are the names a relation can be ``CLASSIFIED_AS``. Falls back to the
    static taxonomy in ``domain_data`` if the graph is unreachable / unclassified.
    """
    try:
        from .clients import neo4j_driver
        with neo4j_driver().session() as s:
            rows = s.run(
                "MATCH (d) WHERE d:LowLevelDomain OR d:MidLevelDomain OR d:HighLevelDomain "
                "AND d.domainName IS NOT NULL RETURN DISTINCT d.domainName AS name"
            ).data()
        names = tuple(r["name"] for r in rows if r.get("name"))
        if names:
            return names
    except Exception as exc:  # pragma: no cover - depends on live DB
        logger.warning("could not load graph domain nodes: %s", exc)
    return tuple(domain_names())


def predict_graph_domains(question: str, analysis: Optional[dict] = None) -> Dict[str, Any]:
    """Predict the GRAPH scope for a question: single / multi / full graph.

    Returns ``{scope, domains, reasoning, keyword_scores}``. ``domains`` are
    validated against the actual domain nodes in the graph; anything unknown or a
    ``full`` scope yields an empty list (→ search the entire graph). This scopes
    ONLY graph retrieval — Hybrid RAG always runs on the full index.
    """
    known = set(graph_domain_nodes())
    descriptions = _domain_descriptions()
    # Only offer the LLM domains that actually exist as nodes (when known).
    offer = [(n, descriptions.get(n, "")) for n in descriptions if not known or n in known]
    domain_block = "\n".join(f"- {name}: {desc}" for name, desc in offer) or \
        "\n".join(f"- {n}" for n in known)

    try:
        data = call_json(
            system_prompt=DOMAIN_PREDICTION_SYSTEM_PROMPT,
            user_prompt=DOMAIN_PREDICTION_USER_PROMPT.format(question=question, domain_block=domain_block),
        )
        pred = validate_model(data, GraphDomainPrediction)
        scope, domains = pred.scope, list(pred.domains)
        reasoning = pred.reasoning
    except Exception as exc:
        logger.warning("predict_graph_domains fallback (%s) → full graph", exc)
        scope, domains, reasoning = "full", [], f"fallback: {exc}"

    # Keep only domains that exist in the graph; drop the rest.
    if known:
        domains = [d for d in domains if d in known]
    if scope == "full" or not domains:
        scope, domains = "full", []
    elif scope == "single" and len(domains) > 1:
        domains = domains[:1]

    return {
        "scope": scope,
        "domains": domains,
        "reasoning": reasoning,
        "keyword_scores": _keyword_scores(question),
    }
