"""LLM-as-judge scoring for the domain-filtering eval answers.

Scores ONE answer to ONE question in isolation (pointwise -- not a side-by-side
with-vs-without comparison; that comparison happens afterwards in _metrics.py by
averaging and diffing these scores across the two conditions). Reuses the app's
existing Azure OpenAI client (src.pipeline.clients.call_json) rather than opening
a new one.

Caveat: there are no gold answers in the eval set (only an expected *domain*
label), so "accuracy" here is a proxy -- internal consistency and apparent
grounding in the cited references -- not verified against the underlying source
text.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from src.pipeline.clients import call_json

logger = logging.getLogger("agent_flow.domain_eval.judge")

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator of answers produced by a tender/ITB \
(Invitation to Bid) question-answering assistant for EPC / oil & gas contracts. \
Score ONE answer to ONE question in isolation -- do not compare it to any other answer. \
Return strict JSON only."""

JUDGE_USER_PROMPT = """Question:
{question}

Answer:
{answer}

Cited sources:
{references_block}

Score the answer 1-5 on each dimension (5 = best):
- completeness: does it cover everything the question asks, without obvious gaps?
- relevance: does it directly address the question, without drifting off-topic?
- fluency: is it clearly written, well-organized, and professional?
- accuracy: is it internally consistent and properly grounded in the cited sources, \
with no unsupported or fabricated-sounding claims?

Return STRICT JSON only:
{{"completeness": <1-5>, "relevance": <1-5>, "fluency": <1-5>, "accuracy": <1-5>, \
"reasoning": "<one to two sentences justifying the scores>"}}
"""

DIMENSIONS = ("completeness", "relevance", "fluency", "accuracy")
ALL_SCORE_KEYS = DIMENSIONS + ("overall",)


def _references_block(references: List[Dict[str, Any]]) -> str:
    if not references:
        return "(none cited)"
    return "\n".join(f"- {r.get('file_name', '?')} (p.{r.get('page_number', '?')})" for r in references)


def judge_answer(question: str, answer: str, references: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One LLM call: score one answer on the four dimensions + a reasoning line."""
    user_prompt = JUDGE_USER_PROMPT.format(
        question=question, answer=answer or "(no answer)",
        references_block=_references_block(references),
    )
    try:
        data = call_json(JUDGE_SYSTEM_PROMPT, user_prompt)
        scores: Dict[str, Any] = {dim: float(data.get(dim, 0)) for dim in DIMENSIONS}
        scores["overall"] = round(sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS), 2)
        scores["reasoning"] = data.get("reasoning", "")
        return scores
    except Exception as exc:
        logger.warning("judge_answer failed: %s", exc)
        return {**{dim: None for dim in ALL_SCORE_KEYS}, "reasoning": f"judge failed: {exc}"}


def judge_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Judge one eval-run row (from _common._run_one), in place. Returns the same dict."""
    if not result.get("ok"):
        result["judge"] = None
        return result
    result["judge"] = judge_answer(result["question"], result.get("answer", ""),
                                   result.get("references") or [])
    return result


def judge_all(results: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """Judge every result in a list, in place, ``max_workers`` at a time."""
    logger.info("Judging %d answers (max_workers=%d)...", len(results), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(judge_result, r): i for i, r in enumerate(results)}
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            logger.info("[judge %d/%d]", done, len(results))
    return results
