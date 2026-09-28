#!/usr/bin/env python3
"""Test script for RAGAS's reference-free context metric.

Deliberately bypasses this pipeline's usual Azure OpenAI / APIM setup --
evaluates via plain OpenAI (OPENAI_API_KEY from .env) instead, using a
separate credential from the rest of the app's LLM calls.

Uses the current (non-deprecated) ragas==0.4.3 API, verified directly against
the installed source after the old SingleTurnSample/LangchainLLMWrapper path
started raising DeprecationWarnings:

  * ragas.metrics.collections.ContextPrecisionWithoutReference -- NOT
    "LLMContextPrecisionWithoutReference". The deprecation warning ragas
    itself prints ("from ragas.metrics.collections import
    LLMContextPrecisionWithoutReference") is wrong -- that module's own
    __init__.py (metrics/collections/context_precision/__init__.py) exports
    it as plain ContextPrecisionWithoutReference. Called as
    ``await metric.ascore(user_input=..., response=..., retrieved_contexts=...)``,
    returning a MetricResult (``.value`` holds the float score) -- not the
    old SingleTurnSample + single_turn_ascore(sample) pattern.

  * ragas.llms.llm_factory(model, provider="openai", client=...) -- replaces
    the deprecated LangchainLLMWrapper, no LangChain wrapping needed at all.
    client must be an ASYNC client (AsyncOpenAI, not OpenAI) -- ascore()
    calls self.llm.agenerate() internally, which raises "Cannot use
    agenerate() with a synchronous client" against a sync client.

Only the reference-free precision metric is used, deliberately -- ragas has
no "recall without reference" metric (LLMContextRecall / NonLLMContextRecall
/ IDBasedContextRecall all require some form of ground truth), and
data/eval_multihop_100_ppr.jsonl's reference_answer is being treated as
unreliable here, so recall is skipped entirely rather than used against a
reference that isn't trusted.

KNOWN COMPATIBILITY ISSUE (found while writing the first version of this
script, not specific to this codebase): ragas==0.4.3's top-level import chain
unconditionally imports `langchain_community.chat_models.vertexai`, which no
longer exists in recent langchain-community releases (>=0.4 has removed
legacy integration shims as that package is sunset). If `import ragas`
raises `ModuleNotFoundError: No module named
'langchain_community.chat_models.vertexai'`, pin an older langchain-community
(e.g. `uv add "langchain-community<0.4"`) or check ragas's current
release notes for the compatible pin.

    cd chatbot
    uv add ragas
    python scripts/domain_eval/test_ragas_metrics.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

CHATBOT_ROOT = Path(__file__).resolve().parents[2]  # .../chatbot
REPO_ROOT = CHATBOT_ROOT.parent
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")  # populates os.environ -- OPENAI_API_KEY lives there

from openai import AsyncOpenAI  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import ContextPrecisionWithoutReference  # noqa: E402

EVAL_MODEL = os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini")

DEFAULT_QUESTION = (
    "How is the combined pipe rack – equipment module connected to the "
    "compressor shelter in the proposed construction sequence?"
)


def get_evaluator_llm():
    """Plain OpenAI (not this app's Azure/APIM setup), authenticated via
    OPENAI_API_KEY. Must be an async client -- see module docstring."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set (checked process env and .env at repo root)")
    async_client = AsyncOpenAI(api_key=api_key)
    return llm_factory(EVAL_MODEL, provider="openai", client=async_client)


async def test_toy_example(evaluator_llm) -> None:
    """The example from the ragas docs -- validates the setup works at all
    before trusting it against real pipeline output."""
    print("=" * 100)
    print("Toy example (Eiffel Tower)")
    print("=" * 100)

    metric = ContextPrecisionWithoutReference(llm=evaluator_llm)
    result = await metric.ascore(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["The Eiffel Tower is located in Paris."],
    )
    print(f"ContextPrecisionWithoutReference: {result.value}")


async def test_real_question(evaluator_llm, question: str) -> None:
    """Run one real question through the PPR + Hybrid pipeline and score its
    actual retrieved contexts + answer."""
    from run_ppr_hybrid_question import run_ppr_hybrid  # same-directory import

    print()
    print("=" * 100)
    print(f"Real pipeline question: {question}")
    print("=" * 100)

    result = run_ppr_hybrid(question)
    answer = result["answer"]
    contexts = [c.get("content", "") for c in result["reranked_chunks"] if c.get("content")]
    print(f"Retrieved contexts: {len(contexts)}")
    print(f"Answer: {answer.answer[:300]}")
    print()

    metric = ContextPrecisionWithoutReference(llm=evaluator_llm)
    score = await metric.ascore(
        user_input=question, response=answer.answer, retrieved_contexts=contexts,
    )
    print(f"ContextPrecisionWithoutReference: {score.value}")


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    evaluator_llm = get_evaluator_llm()
    await test_toy_example(evaluator_llm)
    await test_real_question(evaluator_llm, question)


if __name__ == "__main__":
    asyncio.run(main())
