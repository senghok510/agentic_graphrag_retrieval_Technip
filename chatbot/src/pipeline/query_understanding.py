"""Stage: Query Understanding + Query Type Classification.

Mermaid nodes: ``Query Understanding`` and ``Query Type Classification``.

``analyze_query`` reproduces the notebook routine (HyDE doc, query expansion,
complexity + graph-routing hints, per-sub-question HyDE/expansion). It is the
single LLM-analysis entry point every downstream branch consumes.
``classify_strategy`` maps a question to factoid / relational / summarization
(Simple / Multi-hop / Aggregation) for the non-single-domain graph branch.
"""

from __future__ import annotations

import logging
import re
import time

from .clients import (
    call_json,
    openai_client,
    chat_deployment,
    validate_model,
    canon_entity_categories,
)
from .prompts import (
    HYDE_PROMPT,
    GRAPH_COMPLEXITY_PROMPT,
    QUERY_EXPANSION_PROMPT_COMPLEX,
    QUERY_EXPANSION_PROMPT_SIMPLE,
    STRATEGY_SYSTEM_PROMPT,
    STRATEGY_USER_PROMPT,
    RETRIEVAL_NEED_SYSTEM_PROMPT,
    RETRIEVAL_NEED_USER_PROMPT,
)
from .schemas import (
    GraphComplexityAnalysis,
    StrategyChoice,
    RetrievalNeed,
    ComplexQueryExpansion,
    SimpleQueryExpansion,
)

logger = logging.getLogger("agent_flow.pipeline.query_understanding")

_ANALYST_SYS = (
    "You are a senior specialist in tendering, ITB analysis, EPC contracts, "
    "procurement documents, and RAG query analysis. You do query complexity "
    "analysis and output STRICT JSON only."
)
_HYDE_SYS = (
    "You are a senior specialist in tendering, ITB analysis, EPC contracts, "
    "procurement documents, and technical RAG retrieval."
)
_EXPAND_SYS = (
    "You are a senior specialist in tendering, ITB analysis, EPC contracts, "
    "procurement documents, and RAG retrieval query expansion. "
    "You output STRICT JSON only."
)


def _hyde(question: str) -> str:
    resp = openai_client().chat.completions.create(
        model=chat_deployment(),
        messages=[
            {"role": "system", "content": _HYDE_SYS},
            {"role": "user", "content": HYDE_PROMPT.format(query=question)},
        ],
        temperature=0,
        max_tokens=600,
        timeout=120,
    )
    return (resp.choices[0].message.content or question).strip()


def analyze_query(question: str) -> dict:
    """Understand the query: HyDE doc, expansion, complexity, graph hints.

    Returns the analysis dict consumed by every branch (keys: ``question``,
    ``hyde_doc``, ``expanded_query``, ``expansion_keywords``, ``is_complex``,
    ``sub_questions``, ``entity_hints``, ``category_hints``, ``relation_hints``,
    ``high_level_keywords``, ``hyde_doc_sub_questions``,
    ``expansion_keywords_sub_questions``, ``reasoning``).
    """
    t0 = time.time()
    categories = list(canon_entity_categories())
    try:
        data = call_json(
            system_prompt=_ANALYST_SYS,
            user_prompt=GRAPH_COMPLEXITY_PROMPT.format(
                question=question,
                entity_categories=categories,
            ),
        )
        gca = validate_model(data, GraphComplexityAnalysis)
    except Exception as exc:
        logger.warning("complexity+hints fallback (%s)", exc)
        gca = GraphComplexityAnalysis(
            is_complex=True,
            reasoning="Fallback after complexity and graph-hint parse error.",
            sub_questions=[question],
            entity_hints=[question],
            category_hints=[],
            relation_hints=[],
            high_level_keywords=[],
        )
    logger.info("[analyse] complexity=%s sub=%d — %.1fs",
                gca.is_complex, len(gca.sub_questions), time.time() - t0)

    # ── HyDE for the whole question ──────────────────────────────────────────
    try:
        hyde_doc = _hyde(question)
    except Exception as exc:
        logger.warning("hyde fallback (%s)", exc)
        hyde_doc = question

    # ── Query expansion (keywords + rephrasings) ─────────────────────────────
    expansion_keywords, rephrased_questions = [], []
    expansion_keywords_sub_questions = {}
    sub_questions = []
    if gca.is_complex:
        seen = set()
        sub_questions = [q for q in gca.sub_questions if q and not (q in seen or seen.add(q))]
        if not sub_questions:
            sub_questions = [question]

    try:
        if gca.is_complex:
            data = call_json(_EXPAND_SYS, QUERY_EXPANSION_PROMPT_COMPLEX.format(query=question))
            exp = validate_model(data, ComplexQueryExpansion)
            for sub_question in sub_questions:
                sub_data = call_json(_EXPAND_SYS, QUERY_EXPANSION_PROMPT_COMPLEX.format(query=sub_question))
                exp_sub = validate_model(sub_data, ComplexQueryExpansion)
                expansion_keywords_sub_questions[sub_question] = exp_sub.keywords
        else:
            data = call_json(_EXPAND_SYS, QUERY_EXPANSION_PROMPT_SIMPLE.format(query=question))
            exp = validate_model(data, SimpleQueryExpansion)
        expansion_keywords = exp.keywords or []
        rephrased_questions = getattr(exp, "rephrased_questions", []) or []
    except Exception as exc:
        logger.warning("expansion fallback (%s)", exc)

    expanded_query = (
        f"{question} {' '.join(expansion_keywords)}".strip()
        if expansion_keywords else question
    )

    # ── Per-sub-question HyDE (used by the multi-hop / decomposition path) ────
    hyde_doc_sub_questions = {}
    for sub_question in sub_questions:
        try:
            hyde_doc_sub_questions[sub_question] = _hyde(sub_question)
        except Exception as exc:
            logger.warning("sub-hyde fallback (%s)", exc)
            hyde_doc_sub_questions[sub_question] = sub_question

    # ── Normalise hints ──────────────────────────────────────────────────────
    cats_kept = [c for c in gca.category_hints if c in categories] if categories else list(gca.category_hints)
    relation_hints = [
        re.sub(r"[^a-z0-9]+", "_", r.lower()).strip("_")
        for r in (gca.relation_hints or [])
        if isinstance(r, str) and r.strip()
    ]
    relation_hints = list(dict.fromkeys([r for r in relation_hints if r]))

    return {
        "question": question,
        "original_question": question,
        "hyde_doc": hyde_doc,
        "expansion_keywords_sub_questions": expansion_keywords_sub_questions,
        "hyde_doc_sub_questions": hyde_doc_sub_questions,
        "is_complex": gca.is_complex,
        "expanded_query": expanded_query,
        "expansion_keywords": expansion_keywords,
        "rephrased_questions": rephrased_questions,
        "sub_questions": sub_questions,
        "entity_hints": gca.entity_hints[:10],
        "category_hints": cats_kept,
        "relation_hints": relation_hints,
        "high_level_keywords": list(gca.high_level_keywords or []),
        "reasoning": gca.reasoning,
    }


def classify_retrieval_need(question: str) -> RetrievalNeed:
    """Retrieval Need Classification (v2) → one of the four retrieval algorithms.

    textual_factoid | single_hop | aggregation | multi_hop. This is the HOW-to-retrieve
    axis, independent of domain. Falls back to ``aggregation`` (safest: widest graph +
    hybrid recall) if the LLM call fails.
    """
    try:
        data = call_json(
            system_prompt=RETRIEVAL_NEED_SYSTEM_PROMPT,
            user_prompt=RETRIEVAL_NEED_USER_PROMPT.format(question=question),
        )
        return validate_model(data, RetrievalNeed)
    except Exception as exc:
        logger.warning("retrieval-need fallback (%s)", exc)
        return RetrievalNeed(need="aggregation", query_form="none", reasoning=f"fallback: {exc}")


def classify_strategy(question: str) -> StrategyChoice:
    """(v1, kept for compatibility) Query Type Classification → factoid | relational | summarization."""
    try:
        data = call_json(
            system_prompt=STRATEGY_SYSTEM_PROMPT,
            user_prompt=STRATEGY_USER_PROMPT.format(question=question),
        )
        return validate_model(data, StrategyChoice)
    except Exception as exc:
        logger.warning("strategy fallback (%s)", exc)
        return StrategyChoice(strategy="summarization", reasoning=f"fallback: {exc}")
