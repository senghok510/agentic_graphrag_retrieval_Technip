"""Graph + Vector RAG pipeline (main_project_pipeline.mmd).

Self-contained package that implements the project's retrieval pipeline, built
from the exact LightRAG (evaluate_lightrag.py) and hub-aware PPR (ppr.py) logic
turned into importable functions, organised by mermaid stage:

    query_understanding  Query Understanding + Query Type Classification
    domain_routing       single-domain gate, Domain Prediction, Domain-aware Hybrid Search
    entity_linking       Online Entity Linking (full-text + chunk-mediated + LLM filter)
    retrieval            Semantic Retrieval, Cypher Template Retrieval, hybrid search
    graph_retrieval      shared KG cypher helpers (local 1-hop, semantic relation, chunk windows)
    lightrag             Aggregation branch (LG + SR)
    ppr                  Multi-hop branch (hub-aware Personalized PageRank)
    fusion               Candidate Context Fusion → RRF → Cross-Encoder rerank
    answer / answer_context   grounded, cited LLM answer generation
    orchestrator         the end-to-end flow (run_pipeline)

Entry point: ``from src.pipeline import run_pipeline``.
"""

from .orchestrator import run_pipeline
from .query_understanding import analyze_query, classify_retrieval_need, classify_strategy
from .domain_routing import predict_graph_domains

__all__ = [
    "run_pipeline",
    "analyze_query",
    "classify_retrieval_need",
    "predict_graph_domains",
    "classify_strategy",
]
