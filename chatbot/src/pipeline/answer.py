"""Stage: LLM Answer Generation → Final Grounded Answer.

Mermaid nodes ``LLM`` → ``A``. Turns the reranked candidate chunks into a
numbered, file-grouped context and asks the LLM for a grounded, cited answer,
then remaps citations onto a compact reference list.

``detailed_respond`` is copied verbatim from ppr.py; ``synthesize_answer`` is the
lighter plain-text variant from evaluate_lightrag.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from .clients import call_json, openai_client, chat_deployment
from .prompts import (
    detailed_answer_prompt,
    SYNTHESIZE_ANSWER_SYSTEM_PROMPT,
    SYNTHESIZE_ANSWER_USER_PROMPT,
)
from .answer_context import group_knowledge_base_by_file, build_numbered_context, remap_citations
from .schemas import ResponseGeneration

logger = logging.getLogger("agent_flow.pipeline.answer")


def detailed_respond(analysis: dict, knowledge_base: list, numbered_knowledge_base: str,
                     strategy: str) -> ResponseGeneration:
    """Grounded, cited answer over a numbered, file-grouped knowledge base."""
    total_sources = len(knowledge_base)
    complexity_guidance = ""
    if analysis.get("is_complex") and analysis.get("sub_questions"):
        complexity_guidance = (
            "This is a COMPLEX multi-part query. The sub_questions below help you understand each "
            "sub_question in order to answer efficiently the whole question.\n"
            + "\n".join(f"- {q}" for q in analysis.get("sub_questions", []))
        )

    prompt = detailed_answer_prompt(
        question=analysis["question"], strategy=strategy,
        complexity_guidance=complexity_guidance, total_sources=total_sources,
        numbered_knowledge_base=numbered_knowledge_base,
    )
    data = call_json(system_prompt=prompt, user_prompt="Return STRICT JSON only.")

    answer = data.get("answer", "Error generating detailed response. Please try again.")
    if isinstance(answer, dict):
        answer = answer.get("answer") or answer.get("content") or answer.get("text") \
            or json.dumps(answer, ensure_ascii=False, indent=2)
    elif isinstance(answer, list):
        answer = "\n".join(str(x) for x in answer)
    else:
        answer = str(answer)

    confidence_score = float(data.get("confidence_score", 0.0) or 0.0)
    remapped_answer, references = remap_citations(answer, knowledge_base)
    return ResponseGeneration(answer=remapped_answer, references=references,
                              confidence_score=confidence_score)


def synthesize_answer(question: str, context: str) -> str:
    """Plain-text grounded answer (no citations) — the LightRAG-style variant."""
    resp = openai_client().chat.completions.create(
        model=chat_deployment(),
        messages=[
            {"role": "system", "content": SYNTHESIZE_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": SYNTHESIZE_ANSWER_USER_PROMPT.format(
                question=question, context=context or "N/A")},
        ],
        temperature=0, max_tokens=800, timeout=120,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_answer(analysis: dict, chunks: List[Dict[str, Any]], strategy: str) -> ResponseGeneration:
    """Build the numbered context from reranked chunks and produce the final answer."""
    if not chunks:
        return ResponseGeneration(
            answer="Information not found in available documentation.",
            references=[], confidence_score=0.0)
    knowledge_base = group_knowledge_base_by_file(chunks)
    numbered = build_numbered_context(knowledge_base)
    return detailed_respond(analysis, knowledge_base, numbered, strategy)
