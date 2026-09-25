"""Shared clients and low-level primitives for the retrieval pipeline.

Every pipeline module talks to Azure OpenAI, Azure AI Search and Neo4j through
the helpers defined here, so the notebook-exported logic in ``ppr.py`` and
``evaluate_lightrag.py`` keeps working unchanged while reusing the app's
existing APIM-authenticated clients (``src.config.appSettings``) instead of
re-creating them at import time.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import List

import numpy as np

from ..config.appSettings import (
    get_openai_client,
    embed_document_large_model,
)
from ..services.neo4j_service import get_driver

logger = logging.getLogger("agent_flow.pipeline")

# Default ITB used when the caller does not pass one (mirrors the notebooks).
TARGET_ITB_ID = os.getenv("TARGET_ITB_ID", "ROC_INPEX")

# Embedding config — the KG relation/entity vectors and the AI Search index were
# both built with the 3072-dim text-embedding-3-large deployment behind APIM.
EMBED_DIMENSIONS = 3072
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE", "text-embedding-3-large-llm4t")


def openai_client():
    """APIM-authenticated AzureOpenAI client (chat + embeddings)."""
    return get_openai_client()


def chat_deployment() -> str:
    """Chat completion deployment name (GPT vision deployment behind APIM)."""
    return os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT")


def neo4j_driver():
    """Shared Neo4j driver (pooled by src.services.neo4j_service)."""
    return get_driver()


# ── LLM helpers (verbatim behaviour from the notebooks) ──────────────────────

def call_json(system_prompt: str, user_prompt: str) -> dict:
    """One chat completion constrained to a JSON object, parsed to a dict."""
    resp = openai_client().chat.completions.create(
        model=chat_deployment(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        timeout=120,
    )
    return json.loads(resp.choices[0].message.content)


def call_llm(system_prompt: str, user_prompt: str):
    """Raw JSON-mode completion; returns the full response object."""
    return openai_client().chat.completions.create(
        model=chat_deployment(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=16000,
        timeout=120,
        response_format={"type": "json_object"},
    )


def validate_model(data: dict, model_cls):
    """Pydantic v1/v2 tolerant validation."""
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


# ── Embedding helpers ────────────────────────────────────────────────────────

def embed_texts_large_model(texts: List[str], dimensions: int = EMBED_DIMENSIONS) -> List[List[float]]:
    """Batched embeddings for a list of texts (relations, questions, etc.)."""
    response = openai_client().embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=dimensions,
    )
    return [item.embedding for item in response.data]


def embed_text(text: str, dimensions: int = EMBED_DIMENSIONS) -> List[float]:
    """Single-text embedding (delegates to the app helper)."""
    return embed_document_large_model(text, dimensions=dimensions)


# ── Numeric helpers ──────────────────────────────────────────────────────────

def _l2(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def _minmax(values):
    values = [float(v) for v in values]
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax <= vmin:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


@lru_cache(maxsize=1)
def canon_entity_categories() -> tuple:
    """Distinct canonical entity categories present in the graph.

    Loaded once from Neo4j (the notebooks derive this from ``ENTITY_ROWS`` at
    import). Falls back to an empty tuple if the graph is unreachable so query
    understanding still runs (category hints are simply not constrained).
    """
    try:
        with neo4j_driver().session() as s:
            rows = s.run(
                "MATCH (e:Entity) WHERE e.canonicalCategory IS NOT NULL "
                "RETURN DISTINCT e.canonicalCategory AS c ORDER BY c"
            ).data()
        return tuple(r["c"] for r in rows if r.get("c"))
    except Exception as exc:  # pragma: no cover - depends on live DB
        logger.warning("Could not load canonical entity categories: %s", exc)
        return tuple()
