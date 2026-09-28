"""Branch: Relational / Multi-hop Query (Hub-aware Personalized PageRank).

Mermaid path ``MQ → EL2 → PPR``:
  * Online Entity Linking (EL2) → fused PPR seed entities
  * Hub-aware Personalized PageRank (PPR) → ranked relation subgraph → chunks

Encapsulates ppr.py's module-level notebook cells into functions:
  * ``build_ppr_graph``          — one-time load of the undirected Entity projection + relation meta
  * ``personalized_pagerank``    — equal-seed unweighted PPR (power iteration over the projection)
  * ``score_relations``          — hub-aware relation scoring (degree penalty + bridge bonus)
  * ``run_multihop``             — the branch orchestrator

The relation-scoring maths (degree penalty, bridge bonus) is copied verbatim.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np

from .clients import neo4j_driver
from .entity_linking import fuse_ppr_seeds
from .fusion import rerank_relations
from .graph_retrieval import _attach_graph_weight
from .trace import Stage, noop
from .views import chunk_view, entity_view, relation_view, scored_entity_view

logger = logging.getLogger("agent_flow.pipeline.ppr")

PPR_ALPHA = 0.45  # damping factor (matches the notebook's tuned value)
PPR_MAX_ITER = 100
PPR_TOL = 1e-8
PPR_TOP_ENTITIES = 80
TOP_GRAPH_RELATIONS_FOR_RERANK = 50
TOP_RELATIONS_FOR_CHUNKS = 20

_GRAPH_CACHE: dict | None = None


def build_ppr_graph(refresh: bool = False) -> dict:
    """Load the undirected Entity–Entity projection + relation metadata once.

    Returns a dict with ``entity_meta``, ``adj`` (undirected neighbour sets),
    ``entity_to_relation_ids``, ``relation_meta`` and ``hub_entity_keys``.
    Cached process-wide because the graph is static between ingests.
    """
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None and not refresh:
        return _GRAPH_CACHE

    with neo4j_driver().session() as s:
        graph_rows = [
            r.data()
            for r in s.run("""
            MATCH (subj:Entity)-[:SUBJECT_OF]->(rel:Relation)-[:OBJECT_OF]->(obj:Entity)
            WHERE subj.entityKey IS NOT NULL AND obj.entityKey IS NOT NULL
              AND subj.entityKey <> obj.entityKey
            RETURN subj.entityKey AS source_key, subj.name AS source_name,
                   coalesce(subj.canonicalCategory,'other') AS source_category,
                   obj.entityKey AS target_key, obj.name AS target_name,
                   coalesce(obj.canonicalCategory,'other') AS target_category,
                   rel.relationId AS relation_id,
                   coalesce(rel.relationshipLabel,'related_to') AS relationship_label,
                   coalesce(rel.relationshipDescription,'') AS relationship_description
        """)
        ]
        # relation → domain names (CLASSIFIED_AS). Empty if the graph isn't classified.
        try:
            domain_rows = [
                r.data()
                for r in s.run("""
                MATCH (rel:Relation)-[:CLASSIFIED_AS]->(d)
                WHERE d.domainName IS NOT NULL
                RETURN rel.relationId AS rid, collect(DISTINCT d.domainName) AS domains
            """)
            ]
        except Exception as exc:  # pragma: no cover
            logger.warning("[ppr] CLASSIFIED_AS load failed (%s) — domain scoping disabled", exc)
            domain_rows = []
    relation_domains = {r["rid"]: set(r["domains"]) for r in domain_rows if r.get("rid")}

    entity_meta: dict[str, dict] = {}
    adj: dict[str, set] = defaultdict(set)
    entity_to_relation_ids: dict[str, set] = defaultdict(set)
    relation_meta: dict[str, dict] = {}

    for row in graph_rows:
        sk, tk, rid = row["source_key"], row["target_key"], row["relation_id"]
        entity_meta[sk] = {
            "entity_key": sk,
            "entity_name": row["source_name"],
            "canonical_category": row["source_category"],
        }
        entity_meta[tk] = {
            "entity_key": tk,
            "entity_name": row["target_name"],
            "canonical_category": row["target_category"],
        }
        adj[sk].add(tk)
        adj[tk].add(sk)
        if rid:
            entity_to_relation_ids[sk].add(rid)
            entity_to_relation_ids[tk].add(rid)
            relation_meta[rid] = {
                "relation_id": rid,
                "source_key": sk,
                "source_name": row["source_name"],
                "target_key": tk,
                "target_name": row["target_name"],
                "relationship_label": row["relationship_label"],
                "relationship_description": row["relationship_description"],
            }

    # Hub set: nodes at/above the 99th degree percentile (min 50), as in the notebook.
    degree_values = [len(v) for v in entity_to_relation_ids.values() if len(v) > 0]
    hub_threshold = max(50, int(np.percentile(degree_values, 99))) if degree_values else 50
    hub_entity_keys = {
        k for k, rels in entity_to_relation_ids.items() if len(rels) >= hub_threshold
    }

    _GRAPH_CACHE = {
        "entity_meta": entity_meta,
        "adj": adj,
        "entity_to_relation_ids": entity_to_relation_ids,
        "relation_meta": relation_meta,
        "hub_entity_keys": hub_entity_keys,
        "hub_threshold": hub_threshold,
        "relation_domains": relation_domains,
    }
    logger.info(
        "[ppr] graph loaded: entities=%d relations=%d hubs=%d (threshold=%d) classified=%d",
        len(entity_meta),
        len(relation_meta),
        len(hub_entity_keys),
        hub_threshold,
        len(relation_domains),
    )
    return _GRAPH_CACHE


def _hub_set(entity_to_relation_ids: dict[str, set]) -> tuple:
    """Recompute the hub set + threshold from a (possibly scoped) adjacency."""
    degrees = [len(v) for v in entity_to_relation_ids.values() if len(v) > 0]
    threshold = max(50, int(np.percentile(degrees, 99))) if degrees else 50
    hubs = {k for k, rels in entity_to_relation_ids.items() if len(rels) >= threshold}
    return hubs, threshold


def scoped_graph(graph: dict, domains: list[str] | None, min_relations: int = 5) -> dict:
    """Restrict the PPR graph to relations ``CLASSIFIED_AS`` one of ``domains``.

    Returns a filtered view (adjacency, relation metadata and hub set rebuilt from
    the in-domain relations only). Falls back to the full graph when ``domains`` is
    empty/None, when the graph carries no ``CLASSIFIED_AS`` edges, or when the
    scoped subgraph would be too small to diffuse over (graceful degradation → GENERAL).
    """
    if not domains:
        return graph
    relation_domains = graph.get("relation_domains") or {}
    if not relation_domains:
        return graph
    want = set(domains)
    keep = {rid for rid, ds in relation_domains.items() if ds & want}
    if len(keep) < min_relations:
        logger.info(
            "[ppr] scoped subgraph too small (%d rels) for %s — using full graph",
            len(keep),
            domains,
        )
        return graph

    full_meta = graph["relation_meta"]
    relation_meta = {rid: full_meta[rid] for rid in keep if rid in full_meta}
    adj: dict[str, set] = defaultdict(set)
    entity_to_relation_ids: dict[str, set] = defaultdict(set)
    for rid, meta in relation_meta.items():
        sk, tk = meta["source_key"], meta["target_key"]
        adj[sk].add(tk)
        adj[tk].add(sk)
        entity_to_relation_ids[sk].add(rid)
        entity_to_relation_ids[tk].add(rid)
    hub_entity_keys, hub_threshold = _hub_set(entity_to_relation_ids)
    logger.info(
        "[ppr] scoped to %s: relations=%d entities=%d", domains, len(relation_meta), len(adj)
    )
    return {
        "entity_meta": graph["entity_meta"],
        "adj": adj,
        "entity_to_relation_ids": entity_to_relation_ids,
        "relation_meta": relation_meta,
        "hub_entity_keys": hub_entity_keys,
        "hub_threshold": hub_threshold,
        "relation_domains": relation_domains,
    }


def personalized_pagerank(
    seed_keys: list[str],
    graph: dict,
    alpha: float = PPR_ALPHA,
    max_iter: int = PPR_MAX_ITER,
    tol: float = PPR_TOL,
) -> dict[str, float]:
    """Equal-seed unweighted Personalized PageRank over the entity projection.

    Power iteration on the undirected adjacency with restart to the seed set —
    the portable equivalent of the notebook's GDS ``pageRank.stream`` call
    (``sourceNodes = seeds``, ``dampingFactor = alpha``). Returns a probability
    distribution over reachable entities (sums to 1).
    """
    adj = graph["adj"]
    seeds = [k for k in seed_keys if adj.get(k)]
    if not seeds:
        return {}

    nodes = list(adj.keys())
    index = {k: i for i, k in enumerate(nodes)}
    n = len(nodes)

    restart = np.zeros(n, dtype=np.float64)
    for k in seeds:
        restart[index[k]] = 1.0 / len(seeds)

    rank = restart.copy()
    neighbours = [[index[m] for m in adj[k]] for k in nodes]
    for _ in range(max_iter):
        new_rank = (1.0 - alpha) * restart
        for i in range(n):
            r = rank[i]
            if r == 0.0:
                continue
            nbrs = neighbours[i]
            if not nbrs:
                # dangling node: send mass back to the restart distribution
                new_rank += alpha * r * restart
                continue
            share = alpha * r / len(nbrs)
            for j in nbrs:
                new_rank[j] += share
        if np.abs(new_rank - rank).sum() < tol:
            rank = new_rank
            break
        rank = new_rank

    total = rank.sum() or 1.0
    scores = {nodes[i]: float(rank[i] / total) for i in range(n) if rank[i] > 0}
    top = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:PPR_TOP_ENTITIES])
    return top


def score_relations(entity_scores: dict[str, float], graph: dict) -> dict[str, float]:
    """Hub-aware relation scoring from PPR entity scores (verbatim maths).

    contribution = entity_score / degree (hubs) or / sqrt(degree) (non-hubs);
    final = source_contribution + target_contribution + 0.5 * bridge_bonus.
    """
    relation_meta = graph["relation_meta"]
    entity_to_relation_ids = graph["entity_to_relation_ids"]
    hub_entity_keys = graph["hub_entity_keys"]

    scores = {}
    for relation_id, meta in relation_meta.items():
        sk, tk = meta["source_key"], meta["target_key"]
        source_score = float(entity_scores.get(sk, 0.0))
        target_score = float(entity_scores.get(tk, 0.0))
        source_degree = max(len(entity_to_relation_ids.get(sk, [])), 1)
        target_degree = max(len(entity_to_relation_ids.get(tk, [])), 1)
        source_penalty = source_degree if sk in hub_entity_keys else math.sqrt(source_degree)
        target_penalty = target_degree if tk in hub_entity_keys else math.sqrt(target_degree)
        source_contribution = source_score / source_penalty
        target_contribution = target_score / target_penalty
        bridge_bonus = math.sqrt(source_contribution * target_contribution)
        scores[relation_id] = source_contribution + target_contribution + 0.5 * bridge_bonus
    return scores


def _chunks_for_relations(relation_ids: list[str]) -> list[dict[str, Any]]:
    if not relation_ids:
        return []
    with neo4j_driver().session() as s:
        rows = [
            r.data()
            for r in s.run(
                """
            MATCH (ch:Chunk)-[:ASSERTS]->(rel:Relation)
            WHERE rel.relationId IN $relation_ids
            OPTIONAL MATCH (ch)-[:PART_OF]->(d:Document)
            RETURN ch.ChunkID AS chunk_id,
                   coalesce(ch.text,'') AS content,
                   coalesce(ch.fileName, d.fileName, d.docURL, 'Unknown') AS file_name,
                   coalesce(toString(ch.pageNumber), toString(ch.sequenceNo), 'N/A') AS page_number,
                   collect(DISTINCT rel.relationId) AS relation_ids
        """,
                relation_ids=relation_ids,
            )
        ]
    for r in rows:
        r["source"] = "ppr"
    return rows


def run_multihop(
    analysis: dict,
    vector_rows: list[dict[str, Any]] | None = None,
    domains: list[str] | None = None,
    emit=None,
    tender_id: str | None = None,
) -> dict[str, Any]:
    """Multi-hop branch orchestrator: seeds → (domain-scoped) PPR → relations → chunks.

    Returns ranked chunk rows for Candidate Context Fusion, the top scored
    relations (as graph context), and the resolved seed entities. When
    ``vector_rows`` is None it computes its own small vector seed for the
    chunk-mediated entity-linking channel (so it can run in parallel with Hybrid RAG).
    """
    emit = emit or noop
    if vector_rows is None:
        from .retrieval import vector_seed

        vector_rows = vector_seed(analysis, tender_id=tender_id or None, top=20)

    with Stage(emit, "el2", "Online Entity Linking (PPR seeds)", "fuse_ppr_seeds", "graph") as st:
        seeds = fuse_ppr_seeds(analysis, vector_rows)
        seed_keys = [s.get("entity_key") for s in seeds if s.get("entity_key")]
        st.set(seeds=[s.get("entity_name") for s in seeds], seed_count=len(seed_keys))
        st.attach("entities", "Resolved seed entities", entity_view(seeds))
    if not seed_keys:
        return {"chunks": [], "relations": [], "paths": [], "seeds": []}

    with Stage(
        emit, "ppr", "Hub-aware Personalized PageRank", "personalized_pagerank", "graph"
    ) as st:
        graph = scoped_graph(build_ppr_graph(), domains)
        entity_scores = personalized_pagerank(seed_keys, graph)
        st.set(
            domains=domains or "full",
            graph_entities=len(graph["entity_meta"]),
            graph_relations=len(graph["relation_meta"]),
            hubs=len(graph["hub_entity_keys"]),
            ranked_entities=len(entity_scores),
        )
        st.attach(
            "entities",
            "Top PageRank-ranked entities",
            scored_entity_view(entity_scores, graph["entity_meta"]),
        )
    if not entity_scores:
        return {"chunks": [], "relations": [], "paths": [], "seeds": seeds}

    with Stage(
        emit, "relation_scoring", "Hub-aware Relation Scoring + Rerank", "score_relations", "graph"
    ) as st:
        relation_scores = score_relations(entity_scores, graph)
        top_relation_ids = [
            rid
            for rid, _ in sorted(relation_scores.items(), key=lambda x: x[1], reverse=True)[
                :TOP_GRAPH_RELATIONS_FOR_RERANK
            ]
        ]

        # Assemble relation candidates → cross-encoder rerank against the question.
        relation_meta = graph["relation_meta"]
        candidates = []
        for rid in top_relation_ids:
            meta = relation_meta.get(rid, {})
            if not meta:
                continue
            candidates.append(
                {
                    "relation_id": rid,
                    "subject": meta.get("source_name", "Unknown"),
                    "object": meta.get("target_name", "Unknown"),
                    "label": meta.get("relationship_label", "related_to"),
                    "description": meta.get("relationship_description", ""),
                    "ppr_relation_score": float(relation_scores.get(rid, 0.0)),
                }
            )
        candidates = _attach_graph_weight(candidates)
        reranked = rerank_relations(
            analysis["question"], candidates, top_k=TOP_RELATIONS_FOR_CHUNKS
        )
        st.set(scored_relations=len(relation_scores), top_relations=len(reranked))
        st.attach("relations", "Top hub-aware relations (reranked)", relation_view(reranked))

    with Stage(
        emit, "ppr_chunks", "Relation Evidence Chunks", "_chunks_for_relations", "graph"
    ) as st:
        chunk_rows = _chunks_for_relations([r["relation_id"] for r in reranked])
        st.set(chunks=len(chunk_rows))
        st.attach("chunks", "Chunks asserting the top relations", chunk_view(chunk_rows))
    return {
        "chunks": chunk_rows,
        "relations": reranked,
        "paths": reranked,  # relationship evidence doubles as the "path" context
        "seeds": [s.get("entity_name") for s in seeds],
    }
