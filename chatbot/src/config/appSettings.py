import neo4j
from pydantic_settings import BaseSettings,SettingsConfigDict
from functools import lru_cache
from openai import AzureOpenAI, embeddings
import os
from pathlib import Path
from typing import List
from azure.search.documents import SearchClient
import logging
from azure.core.credentials import AzureKeyCredential
from .APIM_connection import ApimSubscriptionPolicy
import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI

logger=logging.getLogger("agent_flow.apim")

class Settings(BaseSettings):
    AZURE_EMBEDDING_MODEL_LARGE:str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE:str
    AZURE_OPENAI_API_VERSION:str
    AZURE_OPENAI_VISION_DEPLOYMENT:str
    # AZURE_OPENAI_GPT5_MINI_DEPLOYMENT:str
    APIM_SUBSCRIPTION_KEY:str
    AZURE_OPENAI_APIM:str
    AI_SEARCH_APIM:str
    AZURE_SEARCH_INDEX_NAME:str
    API_KEY:str
    NEO4J_URI:str
    NEO4J_USER:str
    NEO4J_PASSWORD:str
    AI_SEARCH_SEMANTIC_SEARCH_CONFIG:str = ""
    
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=Path(__file__).parent.parent.parent / ".env",
        env_file_encoding="utf-8",
    )
    
@lru_cache()
def get_settings():
    return Settings()



_openai_client = None
_langchain_llm = None
def get_openai_client() -> AzureOpenAI:
    """Get or create Azure OpenAI client with APIM configuration"""
    from .appSettings import get_settings
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        if not settings.AZURE_OPENAI_APIM or not settings.APIM_SUBSCRIPTION_KEY:
            raise ValueError("AZURE_OPENAI_APIM and APIM_SUBSCRIPTION_KEY environment variables are required")

        _openai_client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_APIM,
            api_key=settings.API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            default_headers={"Ocp-Apim-Subscription-Key": settings.APIM_SUBSCRIPTION_KEY},
        )

    return _openai_client


def get_langchain_llm(reasoning_effort: str, verbosity: str) -> AzureChatOpenAI :
    """Lazily initialize langchain LLM"""
    from .appSettings import get_settings
    global _langchain_llm
    if _langchain_llm is None:
        try:
            settings = get_settings()
            # _langchain_llm = init_chat_model(
            #     "azure_openai:gpt-4",
            #     azure_deployment=settings.AZURE_OPENAI_GPT5_MINI_DEPLOYMENT,
            #     model_provider="azure-openai",
            # )
            _langchain_llm = AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_VISION_DEPLOYMENT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            api_key=settings.API_KEY,
            default_headers={
                "Ocp-Apim-Subscription-Key": settings.APIM_SUBSCRIPTION_KEY
            },
            azure_endpoint=settings.AZURE_OPENAI_APIM,
            reasoning_effort=reasoning_effort if reasoning_effort else 'medium',
            verbosity=verbosity if verbosity else None
        )
        except Exception as e:
            logging.warning(
                f"Langchain LLM initialization failed, will use direct AzureOpenAI client: {str(e)}"
            )
            _langchain_llm = None
    return _langchain_llm

_llm_instance = None

def get_llm(reasoning_effort: str = 'medium', verbosity: str = None):
    """Get or create the global LLM instance with APIM configuration"""
    global _llm_instance
    if _llm_instance is None:
        llm = get_langchain_llm(reasoning_effort, verbosity)
        if llm is not None:
            _llm_instance = llm
        else:
            logging.warning("Falling back to direct AzureOpenAI client for LLM")
            _llm_instance = get_openai_client()
    return _llm_instance
def embed_document_neo4j(text: str) -> List[float]:
    settings = get_settings()
    params = {
        "model": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE,
        "input": text,
        "dimensions": 384,
    }

    client = get_openai_client()
    response = client.embeddings.create(**params)
    return response.data[0].embedding

def embed_document_large_model(text: str, dimensions: int = 3072) -> List[float]:
    settings = get_settings()
    params = {
            "model": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE,
            "input": text,
            "dimensions": dimensions,
        }

    client = get_openai_client()
    response = client.embeddings.create(**params)
    return response.data[0].embedding

def ai_search_client()-> SearchClient:
    settings = get_settings()
    return SearchClient(
        endpoint=settings.AI_SEARCH_APIM,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential("dummy"),
        per_call_policies=[ApimSubscriptionPolicy(settings.APIM_SUBSCRIPTION_KEY)]
    )