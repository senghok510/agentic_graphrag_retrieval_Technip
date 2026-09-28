# `src/pipeline/` — adaptive retrieval pipeline (v2)

This package **is** the pipeline. It's what `pipeline_app.py` serves (via
`src/routers/pipeline_router.py`) and what the Next.js UI in `web/pipeline-ui/`
streams live.

**Start here:** [`langgraph_pipeline.py`](langgraph_pipeline.py) defines the
shared state, nodes, and compiled graph. [`orchestrator.py`](orchestrator.py)
keeps the existing `run_pipeline(question, ...)` public API as a compatibility
wrapper, so the FastAPI routes, SSE stream, evaluation scripts, and UI do not
need a separate migration.

```python
from src.pipeline import run_pipeline

result = run_pipeline("How are Change Orders related to Re-measurable Works?")
```

## The core flow, and which file owns each step

```
Query Understanding ─────────────────── query_understanding.py (analyze_query)
    ├── Retrieval Need Classification ─ query_understanding.py (classify_retrieval_need)
    │     textual_factoid | single_hop | aggregation | multi_hop
    └── Graph Domain Prediction ─────── domain_routing.py (predict_graph_domains)
          scopes the GRAPH branch only — never Hybrid RAG

Route by retrieval need (langgraph_pipeline.py: GRAPH_BRANCH):
    textual_factoid → Hybrid RAG only ────────────── retrieval.py (hybrid_rag)
    single_hop      → Local Graph Retrieval ──────── lightrag.py (run_single_hop)
    aggregation     → Dual-Level (local + global) ── lightrag.py (run_aggregation)
    multi_hop       → Hub-aware PPR ──────────────── ppr.py (run_multihop)

  LangGraph fans Hybrid RAG and graph retrieval out as parallel nodes and joins
  them at fusion. For textual_factoid, the graph node returns an empty result,
  so only Hybrid RAG contributes candidates.

Evidence Fusion (RRF) ──────────────────────────────── fusion.py (reciprocal_rank_fusion)
Cross-Encoder Reranking (bge-reranker-large) ───────── fusion.py (rerank_chunks)
LLM Answer Generation ───────────────────────────────── answer.py (generate_answer)
```

## File-by-file

### Entry point & event plumbing
| File | Role |
|---|---|
| **`langgraph_pipeline.py`** | **The pipeline graph.** Defines `PipelineState`, each stage node, parallel joins, and the compiled `pipeline_graph`. |
| `orchestrator.py` | Compatibility entry point. Its `run_pipeline()` delegates to the compiled LangGraph while preserving the legacy result and callback contracts. |
| `__init__.py` | Public API: re-exports `run_pipeline`, `analyze_query`, `classify_retrieval_need`, `predict_graph_domains`, `classify_strategy`. Everything not listed here is an internal implementation detail of some stage. |
| `streaming.py` | SSE transport adapter — runs `run_pipeline` in a worker thread and forwards each stage event to the FastAPI response as `data: {...}\n\n` frames. Not pipeline *logic*; purely how the UI watches it live. |
| `trace.py` | The `Stage` context manager every stage below uses to emit `start`/`done`/`error` events and measure elapsed time. A no-op with zero overhead if no `emit` callback is passed (e.g. in `scripts/domain_eval/` runner scripts). |

### Stage 1 — Query Understanding & Routing (always runs, decides everything downstream)
| File | Role |
|---|---|
| `query_understanding.py` | `analyze_query` (HyDE doc, query expansion, entity/category hints, sub-questions) and `classify_retrieval_need` (textual_factoid / single_hop / aggregation / multi_hop). |
| `domain_routing.py` | `predict_graph_domains` — keyword-vote + LLM prediction against the 3-tier domain taxonomy (`domain_data/`). Scopes the graph branch's search space only; Hybrid RAG always searches the full index regardless. |

### Stage 2 — Retrieval branches (the actual retrieval *methods*)
| File | Role |
|---|---|
| `retrieval.py` | **Hybrid RAG** — `hybrid_rag` (BM25 + dense vector against the full Azure AI Search index). Runs on every route; the one thing every question retrieves through. |
| `lightrag.py` | **LightRAG-style graph retrieval** — `run_single_hop` (1-hop local neighborhood) and `run_aggregation` (local + global dual-level). Ports `evaluate_lightrag.py`'s logic into callable functions. |
| `ppr.py` | **Hub-aware Personalized PageRank** — `run_multihop`. Builds an in-memory entity graph (`build_ppr_graph`), runs PPR from linked seed entities, scores/reranks the resulting relations. Ports the root `ppr.py` notebook logic. Computes everything in-memory — never writes `PPR_REL` back to Neo4j (that's only done by the legacy exploratory notebooks). |
| `entity_linking.py` | Shared entity-resolution cascade — `link_entities` (BM25 → chunk-mediated → lexical rank → LLM match filter → LLM dedupe) and `fuse_ppr_seeds`. Called by **both** `lightrag.py` and `ppr.py` to turn question text into KG entity keys; not a branch on its own. |
| `graph_retrieval.py` | Shared low-level Cypher that `lightrag.py` and `ppr.py` both build on: `_local_candidate_relations` (1-hop), `_global_search` (keyword→relation vectors), `_relation_window_chunks` (relation→chunk widening via `NEXT`), `_attach_graph_weight`. |

### Stage 3 — Fusion & Reranking (always runs, merges whatever the branches returned)
| File | Role |
|---|---|
| `fusion.py` | `reciprocal_rank_fusion` (merge/dedupe candidate chunk lists), `rerank_chunks` / `rerank_relations` (cross-encoder `BAAI/bge-reranker-large`). Also owns the reranker's process-wide singleton (`get_reranker()` / `warmup_reranker()`, loaded once, thread-safe). |

### Stage 4 — Answer Generation (always runs, last step)
| File | Role |
|---|---|
| `answer.py` | `generate_answer` — turns the final reranked chunks into a grounded, cited `ResponseGeneration`. |
| `answer_context.py` | Shared helpers `answer.py` *and* the retrieval branches use: chunk evidence → numbered file-grouped sources, citation remapping, whitespace cleanup. Split out to avoid an import cycle. |

### Supporting infrastructure (not a pipeline stage, but everything above depends on it)
| File | Role |
|---|---|
| `clients.py` | Lazy, cached Azure OpenAI / Azure AI Search / Neo4j client wrappers. Every stage talks to external services through here — it deliberately reuses `src/services`' connections rather than opening new ones, so the whole app shares one connection pool and one APIM auth path. |
| `schemas.py` | Shared Pydantic models (`ResponseGeneration`, `ReferenceGeneration`, query-expansion models) used by the pipeline and the FastAPI layer alike. |
| `prompts.py` | Every LLM prompt template used anywhere in the pipeline, centralized in one file so behavior stays in sync across stages. |
| `views.py` | Truncates/bounds internal rows (chunks, relations, entities) into small JSON for the SSE stage payloads. UI-facing only — never touches what actually gets returned to `run_pipeline`'s caller or fed to the LLM. |

## Tracing one real request

For a `multi_hop` question, the file-level call path is:

```
orchestrator.run_pipeline → langgraph_pipeline.pipeline_graph
 ├─ query_understanding.analyze_query
 ├─ query_understanding.classify_retrieval_need   → "multi_hop"
 ├─ domain_routing.predict_graph_domains
 ├─ (concurrently)
 │   ├─ retrieval.hybrid_rag
 │   └─ ppr.run_multihop
 │        ├─ entity_linking.fuse_ppr_seeds
 │        ├─ graph_retrieval.build_ppr_graph / scoped_graph   (via clients.neo4j_driver)
 │        ├─ fusion.rerank_relations
 │        └─ graph_retrieval._relation_window_chunks
 ├─ fusion.reciprocal_rank_fusion
 ├─ fusion.rerank_chunks
 └─ answer.generate_answer → answer_context helpers → schemas.ResponseGeneration
```

`single_hop` and `aggregation` follow the same shape, swapping `ppr.py` for
`lightrag.run_single_hop` / `lightrag.run_aggregation`. `textual_factoid` skips
the graph branch entirely — just `retrieval.hybrid_rag` straight into fusion.
