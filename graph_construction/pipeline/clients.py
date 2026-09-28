"""Lazy, thread-safe, cached clients (Azure OpenAI, Azure AI Search, Neo4j) +
the retryable LLM call and embedding helpers built on top of them.

Nothing here connects to anything at import time -- each get_*() only builds
its client on first use, guarded by a lock (the pipeline runs these from many
worker threads concurrently via ThreadPoolExecutor).
"""

from __future__ import annotations

import logging
import os
import re
import threading

from azure.core.credentials import AccessToken, AzureKeyCredential
from azure.core.pipeline.policies import SansIOHTTPPolicy
from azure.search.documents import SearchClient
from neo4j import GraphDatabase
from openai import APIConnectionError, APITimeoutError, AzureOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from . import config

logger = logging.getLogger("graph_construction.pipeline.clients")


class ApimSubscriptionPolicy(SansIOHTTPPolicy):
    """Custom policy for APIM Subscription Key Authentication."""

    def __init__(self, subscription_key: str):
        super().__init__()
        self.subscription_key = subscription_key.strip() if subscription_key else ""
        self.operation_id = None

    def on_request(self, request):
        if not self.subscription_key:
            logger.warning("APIM subscription key is empty")
            return
        if "authorization" in request.http_request.headers:
            del request.http_request.headers["authorization"]
        request.http_request.headers["Ocp-Apim-Subscription-Key"] = self.subscription_key

    def on_response(self, request, response):
        if response.http_response.status_code >= 400:
            logger.error("APIM Response Error: %s", response.http_response.status_code)
            if hasattr(response.http_response, "text"):
                text = response.http_response.text
                body = text() if callable(text) else text
                if body:
                    logger.error("Response body: %s", body[:500])
        if "Operation-Location" in response.http_response.headers:
            operation_location = response.http_response.headers["Operation-Location"]
            match = re.search(r"/analyzeResults/([a-f0-9\-]{36})", operation_location)
            if match:
                self.operation_id = match.group(1)


class NoOpCredential:
    """Dummy credential for APIM-based authentication."""

    def get_token(self, *args, **kwargs):
        return AccessToken(os.getenv("API_KEY", ""), 0)


# ── Lazy, thread-safe singletons ──────────────────────────────────────────────

_openai_client: AzureOpenAI | None = None
_openai_lock = threading.Lock()

_search_client: SearchClient | None = None
_search_lock = threading.Lock()

_neo4j_driver = None
_neo4j_lock = threading.Lock()


def get_openai_client() -> AzureOpenAI:
    global _openai_client
    if _openai_client is None:
        with _openai_lock:
            if _openai_client is None:
                _openai_client = AzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_APIM"),
                    api_key="DUMMY",  # auth happens via the APIM subscription header, not this
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                    default_headers={
                        "Ocp-Apim-Subscription-Key": os.getenv("APIM_SUBSCRIPTION_KEY")
                    },
                )
    return _openai_client


def get_search_client() -> SearchClient:
    global _search_client
    if _search_client is None:
        with _search_lock:
            if _search_client is None:
                _search_client = SearchClient(
                    endpoint=os.getenv("AI_SEARCH_APIM"),
                    index_name=config.SEARCH_INDEX_NAME,
                    credential=AzureKeyCredential("DUMMY"),
                    per_call_policies=[ApimSubscriptionPolicy(os.getenv("APIM_SUBSCRIPTION_KEY"))],
                )
    return _search_client


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        with _neo4j_lock:
            if _neo4j_driver is None:
                _neo4j_driver = GraphDatabase.driver(
                    os.getenv("NEO4J_URI"),
                    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
                    connection_timeout=10,
                    max_connection_lifetime=600,
                )
    return _neo4j_driver


def chat_deployment() -> str:
    return os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT")


# ── LLM + embedding helpers ────────────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
    wait=wait_exponential(min=2, max=60),
    stop=stop_after_attempt(5),
)
def call_llm(system_prompt: str, user_prompt: str):
    """One chat completion, JSON-mode, with retry on transient API errors."""
    return get_openai_client().chat.completions.create(
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


def embed_texts_large_model(
    texts: list[str], dimensions: int = config.EMBED_DIMENSIONS
) -> list[list[float]]:
    response = get_openai_client().embeddings.create(
        model=config.EMBED_MODEL,
        input=texts,
        dimensions=dimensions,
    )
    return [item.embedding for item in response.data]
