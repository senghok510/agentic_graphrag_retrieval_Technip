"""Shared runner for the LightRAG-only domain-filtering comparison.

Bypasses Hybrid RAG and the retrieval-need router entirely -- every question is
forced through LightRAG (run_aggregation) regardless of what
classify_retrieval_need would normally pick, and Hybrid RAG never runs at all.
This isolates domain filtering's effect on retrieval: Hybrid RAG always
searches the full index regardless of domain scope (by design -- see
orchestrator.py's docstring), so including it in a with/without-domain-
filtering comparison would dilute the signal with a component that behaves
identically either way, which wouldn't be a fair comparison.

Two domain-scoping conditions, both feeding the SAME LightRAG retrieval path:
  * baseline     -- domains=None (no filter at all; today's LightRAG code,
                     just with scoping turned off -- "like prod" minus the
                     domain step)
  * single_multi -- domains from a single LLM call over the FULL domain list
                     (no retrieval/RRF stage). The LLM must produce a
                     "domain_scan" entry (fits true/false + a short note) for
                     every domain, in order, before it's allowed to commit to
                     a final answer -- since call_json uses strict JSON mode
                     (no free text outside the JSON object is possible), this
                     is how a systematic per-domain comparison gets forced
                     while staying inside that constraint: ordering the
                     schema so the scan has to be generated first,
                     autoregressively, before the final scope/domains fields.
                     Ported from scripts/domain_eval/run_comparison_lightrag_only.ipynb,
                     which is the actual place to keep iterating on this
                     prompt -- port changes back here once validated there.

Produces result rows in the same shape _common._run_one already produces, so
_llm_judge.py and _metrics.py are reused completely unchanged.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

CHATBOT_ROOT = Path(__file__).resolve().parents[2]  # .../chatbot
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from src.pipeline.query_understanding import analyze_query  # noqa: E402
from src.pipeline.lightrag import run_aggregation  # noqa: E402
from src.pipeline.fusion import reciprocal_rank_fusion, rerank_chunks  # noqa: E402
from src.pipeline.answer import generate_answer  # noqa: E402
from src.pipeline.clients import call_json, validate_model  # noqa: E402

logger = logging.getLogger("agent_flow.domain_eval.lightrag_only")

DOMAIN_DATA_DIR = CHATBOT_ROOT / "domain_data"
QUESTIONS_PATH = DOMAIN_DATA_DIR / "domain_eval_questions.json"
RESULTS_DIR = CHATBOT_ROOT / "domain_data" / "eval_results"

RRF_POOL = 50
FINAL_CONTEXT_CHUNKS = 20
MAX_MULTI_DOMAINS = 2


def load_questions(limit: Optional[int] = None) -> List[Dict[str, str]]:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    qs = data["questions"]
    return qs[:limit] if limit else qs


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d results -> %s", len(results), path)


# ── Single|multi domain verifier: LLM-only, full domain list, no retrieval
# stage. Kept in sync with scripts/domain_eval/run_comparison_lightrag_only.ipynb
# so this module doesn't depend on the notebook, but the notebook is where
# prompt iteration actually happens ────────────────────────────────────────

@lru_cache(maxsize=1)
def _domain_descriptions() -> Dict[str, str]:
    return json.loads((DOMAIN_DATA_DIR / "domain_description_ROC_INPEX.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _domain_keywords() -> Dict[str, List[str]]:
    return json.loads((DOMAIN_DATA_DIR / "domain_keywords_ROC_INPEX.json").read_text(encoding="utf-8"))


def _domain_block() -> str:
    descriptions = _domain_descriptions()
    keywords = _domain_keywords()
    lines = []
    for name, desc in descriptions.items():
        kw = keywords.get(name) or []
        kw_str = f" (keywords: {', '.join(kw)})" if kw else ""
        lines.append(f"- {name}: {desc}{kw_str}")
    return "\n".join(lines)


class DomainScanEntry(BaseModel):
    domain: str
    fits: bool
    note: str = ""


class SingleMultiVerification(BaseModel):
    domain_scan: List[DomainScanEntry] = []
    scope: Literal["single", "multi"]
    domains: List[str]
    reasoning: str = ""


_VERIFIER_SYS = (
    "You are a senior engineering document classifier for EPC (Engineering, Procurement, "
    "Construction) tender and ITB (Invitation to Bid) documents. You determine which "
    "discipline domain(s) a question falls under, drawing on deep familiarity with process "
    "and detailed engineering, procurement, construction, project management, and the "
    "commercial/legal practices of tendering. Respond with strict JSON only -- no prose, no "
    "markdown, and no explanation outside the JSON object."
)

_VERIFIER_USER = """Classify the question below into the domain(s) it belongs to, using ONLY the
domain list provided below -- this is the complete list of available domains, not a shortlist.

Process to follow (do this before deciding):
1. Go through the domain list below ONE BY ONE, in order. For each domain, read its description
   and keywords and check whether the question's CORE SUBJECT -- the specific equipment, system,
   or discipline it is actually about -- matches that domain. Record a brief true/false + one-line
   note for every domain in "domain_scan".
2. Do not classify based on a generic action/aspect word alone (e.g. "design requirements",
   "procedures", "specifications", "testing", "management" appear across nearly every domain and
   carry no discriminating signal by themselves) -- match the SUBJECT, not the verb.
3. Do not classify a domain just because a word in the question happens to match or resemble that
   domain's NAME -- check the domain's full description, not a name-level word coincidence. A
   question about "quality sample analysis in a laboratory" is about LABORATORY (physical testing
   facilities, sample handling), not QUALITY MANAGEMENT (QA/QC processes, audits, certification),
   even though the word "quality" appears in the question.
4. Only after completing the scan, decide:
   - scope = "single": exactly one domain clearly fits the core subject.
   - scope = "multi": answering the question fully genuinely requires combining information from
     more than one domain (up to {max_multi}) -- not because a term loosely touches another domain
     in passing. Prefer "single" when in doubt; use "multi" only when it is clearly warranted.
- This question is already confirmed to be in scope for one of these domains -- you must always
  select at least one domain; never return an empty list or decline to classify.
- Each entry in "domains" must be copied EXACTLY as it appears below (same spelling, spacing, and
  punctuation) -- any name that does not match exactly will be discarded by the calling code.

Examples of correct final reasoning (illustrative domain names only -- always classify against the
actual domain list given to you below, not the names used in these examples; your actual answer
must also include a full "domain_scan" covering every domain listed below, omitted here for
brevity):

Question: "What are the design requirements for the fired heaters in the process package?"
Reasoning: the core subject is "fired heaters" -- a specific, named equipment type -- not the
generic phrase "design requirements" or the word "process".
Correct final fields: {{"scope": "single", "domains": ["FIRED EQUIPMENT"], "reasoning": "The question is specifically about fired heaters, a named equipment type, not a general process-design topic."}}

Question: "What testing and inspection procedures apply to pressure vessel fabrication?"
Reasoning: the core subject is "pressure vessel fabrication" -- not "testing and inspection
procedures" alone, which is a generic aspect that could apply to almost any equipment.
Correct final fields: {{"scope": "single", "domains": ["PRESSURE VESSELS"], "reasoning": "Fabrication and testing here concern pressure vessels specifically, not quality management in general."}}

Question: "What vendor qualification criteria and delivery schedule commitments apply to rotating equipment packages?"
Reasoning: this genuinely spans two domains -- vendor qualification and delivery are procurement
concerns, and rotating equipment is its own engineering discipline -- neither answers the question
alone.
Correct final fields: {{"scope": "multi", "domains": ["PROCUREMENT", "ROTATING EQUIPMENT"], "reasoning": "The question asks about both vendor/procurement terms and rotating-equipment-specific requirements."}}

Question: "What laboratory testing facilities are required for quality sample analysis?"
Reasoning: the core subject is "laboratory testing facilities" -- the word "quality" here describes
the PURPOSE of the sample analysis, it does not signal the QUALITY MANAGEMENT domain (which covers
QA/QC processes, audits, and certification programs -- not physical lab facilities or equipment).
Correct final fields: {{"scope": "single", "domains": ["LABORATORY"], "reasoning": "The question asks about physical laboratory testing facilities, not quality-assurance processes or audits."}}

Available domains (name: description (keywords)):
{domain_block}

Question: {question}

Return STRICT JSON in exactly this shape, with no additional keys. "domain_scan" must contain
exactly one entry per domain listed above, in the same order:
{{"domain_scan": [{{"domain": "<exact domain name>", "fits": true|false, "note": "<very short reason, a few words>"}}, ...], "scope": "single|multi", "domains": ["<exact domain name from the list above>", ...], "reasoning": "<one concise sentence citing the specific part of the question that justifies this choice>"}}
"""


def predict_single_multi(question: str) -> Dict[str, Any]:
    """LLM-only domain prediction: no retrieval/fusion stage -- the LLM scans every
    domain's name + description + keywords one by one (via the "domain_scan" field,
    generated before the final answer within the same JSON call) and only then decides.

    Note: this makes each call noticeably longer (one domain_scan entry per domain,
    ~40 domains) than a direct-answer prompt would be -- more latency/cost per
    question in exchange for forcing the systematic per-domain comparison.
    """
    all_domains = list(_domain_descriptions().keys())
    domain_block = _domain_block()

    domain_scan: List[Dict[str, Any]] = []
    try:
        data = call_json(_VERIFIER_SYS, _VERIFIER_USER.format(
            domain_block=domain_block, question=question, max_multi=MAX_MULTI_DOMAINS))
        pred = validate_model(data, SingleMultiVerification)
        scope, domains, reasoning = pred.scope, list(pred.domains), pred.reasoning
        domain_scan = [e.model_dump() if hasattr(e, "model_dump") else e.dict() for e in pred.domain_scan]
    except Exception as exc:
        # No retrieval ranking to fall back on here -- there's no signal for which
        # domain is most plausible if the LLM call itself fails, so this just picks
        # the first domain in the description file, which is an arbitrary guess,
        # not a ranked one. Worth knowing if fallback rate is high.
        logger.warning("predict_single_multi (LLM-only) fallback (%s) -> arbitrary first domain", exc)
        scope, domains, reasoning = "single", all_domains[:1], f"fallback: {exc}"

    domains = [d for d in domains if d in all_domains]
    if not domains:
        scope, domains = "single", all_domains[:1]
    elif scope == "single" and len(domains) > 1:
        domains = domains[:1]
    elif scope == "multi":
        domains = domains[:MAX_MULTI_DOMAINS]

    return {"scope": scope, "domains": domains, "reasoning": reasoning,
            "top_candidates": all_domains, "domain_scan": domain_scan}


# ── LightRAG-only pipeline: Query Understanding -> LightRAG -> RRF -> rerank -> answer ──

def run_lightrag_only(question: str, use_domain_filter: bool) -> Dict[str, Any]:
    analysis = analyze_query(question)

    domain_pred: Optional[Dict[str, Any]] = None
    domains: Optional[List[str]] = None
    if use_domain_filter:
        domain_pred = predict_single_multi(question)
        domains = domain_pred["domains"] or None

    graph_result = run_aggregation(analysis, domains=domains)
    graph_chunks = graph_result.get("chunks", [])

    # Single-source RRF (no Hybrid RAG to fuse against) -- kept for consistency
    # with the RRF-then-rerank convention the rest of the pipeline uses.
    fused = reciprocal_rank_fusion([graph_chunks], top_k=RRF_POOL)
    reranked = rerank_chunks(question, fused, top_k=FINAL_CONTEXT_CHUNKS)

    answer = generate_answer(analysis, reranked, strategy="aggregation")

    return {
        "answer": answer,
        "domain_pred": domain_pred,
        "domains": domains,
        "graph_chunks": len(graph_chunks),
        "fused": len(fused),
        "reranked": len(reranked),
    }


def _run_one(item: Dict[str, str], use_domain_filter: bool) -> Dict[str, Any]:
    t0 = time.time()
    try:
        result = run_lightrag_only(item["question"], use_domain_filter=use_domain_filter)
        answer = result["answer"]
        domain_pred = result["domain_pred"]
        return {
            "id": item["id"],
            "expected_domain": item["domain"],
            "question": item["question"],
            "ok": True,
            "route": "aggregation",  # LightRAG only, always
            # "full" here means "unscoped" in _metrics.py's vocabulary -- keep the
            # exact string so the reused compare()/print_report() logic (which
            # checks domain_scope == "full") buckets the baseline correctly
            # instead of wrongly scoring it as a domain-prediction miss.
            "domain_scope": (domain_pred["scope"] if domain_pred else "full"),
            "predicted_domains": (domain_pred["domains"] if domain_pred else []),
            "graph_domains": result["domains"],
            "domain_filter_disabled": not use_domain_filter,
            # [graph_chunks, 0] rather than just [graph_chunks]: _metrics._graph_chunk_count
            # requires len(candidate_counts) > 1 (it expects [graph, hybrid] like the full
            # orchestrator produces) -- the 0 stands in for "0 hybrid chunks: Hybrid RAG
            # never ran" and is literally true, not a placeholder.
            "candidate_counts": [result["graph_chunks"], 0],
            "fused": result["fused"],
            "reranked": result["reranked"],
            "confidence": answer.confidence_score,
            "answer": answer.answer,
            "references": [r.model_dump() if hasattr(r, "model_dump") else r for r in answer.references],
            "elapsed_s": round(time.time() - t0, 2),
        }
    except Exception as exc:
        logger.exception("run_lightrag_only failed for %s", item["id"])
        return {
            "id": item["id"],
            "expected_domain": item["domain"],
            "question": item["question"],
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.time() - t0, 2),
        }


def run_both_conditions(questions: List[Dict[str, str]],
                        max_workers: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run every question under BOTH conditions in one shared thread pool.

    Returns (single_multi_results, baseline_results), each in question order.
    """
    id_to_index = {item["id"]: i for i, item in enumerate(questions)}
    tasks: List[Tuple[Dict[str, str], bool]] = (
        [(item, True) for item in questions] + [(item, False) for item in questions]
    )

    filtered_results: List[Optional[Dict[str, Any]]] = [None] * len(questions)
    baseline_results: List[Optional[Dict[str, Any]]] = [None] * len(questions)

    logger.info("Running %d questions x 2 conditions = %d LightRAG-only pipeline calls "
               "(max_workers=%d)...", len(questions), len(tasks), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_one, item, use_filter): (item, use_filter)
            for item, use_filter in tasks
        }
        done = 0
        for future in as_completed(futures):
            item, use_filter = futures[future]
            idx = id_to_index[item["id"]]
            result = future.result()
            (filtered_results if use_filter else baseline_results)[idx] = result
            done += 1
            cond = "single|multi filter" if use_filter else "baseline (no filter)"
            logger.info("[%d/%d] %s (%s)", done, len(tasks), item["id"], cond)

    return filtered_results, baseline_results  # type: ignore[return-value]
