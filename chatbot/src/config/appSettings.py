"""Application settings and lazily initialized external-service clients.

The filename is retained for compatibility with existing imports. New code
should consume the functions in this module rather than creating SDK clients
directly.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

from .APIM_connection import ApimSubscriptionPolicy

logger = logging.getLogger("agent_flow.apim")

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Validated configuration loaded from environment variables or `.env`."""

    AZURE_EMBEDDING_MODEL_LARGE: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_VISION_DEPLOYMENT: str
    APIM_SUBSCRIPTION_KEY: str
    AZURE_OPENAI_APIM: str
    AI_SEARCH_APIM: str
    AZURE_SEARCH_INDEX_NAME: str
    API_KEY: str
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    AI_SEARCH_SEMANTIC_SEARCH_CONFIG: str = ""

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_openai_client() -> AzureOpenAI:
    """Return the process-wide APIM-authenticated Azure OpenAI client."""

    settings = get_settings()
    return AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_APIM,
        api_key=settings.API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        default_headers={"Ocp-Apim-Subscription-Key": settings.APIM_SUBSCRIPTION_KEY},
    )


@lru_cache(maxsize=8)
def get_langchain_llm(
    reasoning_effort: str = "medium",
    verbosity: str | None = None,
) -> AzureChatOpenAI | None:
    """Return a cached LangChain model configured for the requested profile."""

    settings = get_settings()
    try:
        return AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_VISION_DEPLOYMENT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            api_key=settings.API_KEY,
            default_headers={
                "Ocp-Apim-Subscription-Key": settings.APIM_SUBSCRIPTION_KEY,
            },
            azure_endpoint=settings.AZURE_OPENAI_APIM,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )
    except Exception as exc:  # pragma: no cover - depends on installed SDK/model
        logger.warning("LangChain model initialization failed: %s", exc)
        return None


def get_llm(reasoning_effort: str = "medium", verbosity: str | None = None):
    """Return the LangChain model, falling back to the direct OpenAI client."""

    return get_langchain_llm(reasoning_effort, verbosity) or get_openai_client()


def _embed(text: str, dimensions: int) -> list[float]:
    settings = get_settings()
    response = get_openai_client().embeddings.create(
        model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE,
        input=text,
        dimensions=dimensions,
    )
    return response.data[0].embedding


def embed_document_neo4j(text: str) -> list[float]:
    return _embed(text, dimensions=384)


def embed_document_large_model(text: str, dimensions: int = 3072) -> list[float]:
    return _embed(text, dimensions=dimensions)


@lru_cache(maxsize=1)
def ai_search_client() -> SearchClient:
    """Return the process-wide Azure AI Search client."""

    settings = get_settings()
    return SearchClient(
        endpoint=settings.AI_SEARCH_APIM,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential("dummy"),
        per_call_policies=[ApimSubscriptionPolicy(settings.APIM_SUBSCRIPTION_KEY)],
    )
