"""Shared graph-retrieval Cypher helpers.

The low-level KG reads used by both the Aggregation (LightRAG) and Multi-hop
(PPR) branches:
  * ``_local_candidate_relations`` — 1-hop relations of seed entities (Local Graph Retrieval)
  * ``_global_search``             — high-level keywords → relation vectors (Semantic Relation Retrieval)
  * ``_relation_window_chunks``    — chunks that ASSERT relations, widened via NEXT
  * ``_attach_graph_weight``       — uniform graph weight for fair rerank blending

Copied verbatim from ``hybrid_lh_agent.py`` / ``evaluate_lightrag.py``, retargeted
to the shared pipeline Neo4j driver.
"""

from __future__ import annotations

from typing import Any

from .clients import neo4j_driver

# Domain scope filter (v2): keep a relation only if it is CLASSIFIED_AS one of the
# predicted domains. ``$domains = null`` (full graph) disables the filter entirely.
# Uses a pattern comprehension so it works on Neo4j 4.x+ without EXISTS subqueries.
_DOMAIN_FILTER = (
    "($domains IS NULL OR "
    "size([(r)-[:CLASSIFIED_AS]->(dom) WHERE dom.domainName IN $domains | dom]) > 0)"
)


def _local_candidate_relations(
    entity_keys: list[str], pool: int = 120, src_per_rel: int = 3, domains: list[str] | None = None
) -> list[dict[str, Any]]:
    """All 1-hop relations of the seed entities, graph-weight ranked, bounded IN-DB.

    Each relation carries its ASSERTS source chunks (chunk_id, text, file_name,
    page_number). Hubs never ship their full edge-star to Python — ``LIMIT $pool``
    is applied inside Cypher after ranking. When ``domains`` is given, relations are
    restricted to those ``CLASSIFIED_AS`` one of those domains (graph domain scoping).
    """
    if not entity_keys:
        return []
    with neo4j_driver().session() as sess:
        return sess.run(
            f"""
            UNWIND $keys AS ekey
            MATCH (seed:Entity {{entityKey: ekey}})
            MATCH (sub:Entity)-[:SUBJECT_OF]->(r:Relation)-[:OBJECT_OF]->(obj:Entity)
            WHERE (sub = seed OR obj = seed) AND {_DOMAIN_FILTER}
            WITH DISTINCT r, sub, obj,
                 coalesce(sub.relDegree,0) + coalesce(obj.relDegree,0) AS edge_rank,
                 coalesce(r.aggregatedWeight, r.weight, r.relationshipStrength/10.0, 0) AS weight
            ORDER BY weight DESC, edge_rank DESC
            LIMIT $pool

            OPTIONAL MATCH (ch:Chunk)-[:ASSERTS]->(r)
            WITH r, sub, obj, edge_rank, weight,
                 collect(DISTINCT ch)[..$srck] AS chs

            RETURN r.relationId AS relation_id, sub.name AS subject, obj.name AS object,
                   r.relationshipLabel AS label, r.relationshipDescription AS description,
                   edge_rank, weight,
                   [c IN chs | {{
                       chunk_id:    c.ChunkID,
                       text:        c.text,
                       file_name:   c.fileName,
                       page_number: c.pageNumber
                   }}] AS sources
        """,
            {"keys": entity_keys, "pool": pool, "srck": src_per_rel, "domains": domains or None},
        ).data()


def _global_search(
    hl_keywords, top_k: int = 20, domains: list[str] | None = None
) -> list[dict[str, Any]]:
    """High-level theme keywords → relation vector search → relations + endpoints + chunks.

    When ``domains`` is given, results are restricted to relations ``CLASSIFIED_AS``
    one of those domains; the vector query is widened first so the post-filter still
    returns a useful set.
    """
    hl_keywords = list(hl_keywords or [])
    if not hl_keywords:
        return []
    from .clients import embed_texts_large_model

    qvec = embed_texts_large_model([", ".join(hl_keywords)])[0]
    # widen recall before the domain filter cuts the result set
    query_k = top_k * 4 if domains else top_k
    with neo4j_driver().session() as session:
        return session.run(
            f"""
            CALL db.index.vector.queryNodes('relation_embedding_index', $query_k, $qvec)
            YIELD node AS r, score
            WHERE {_DOMAIN_FILTER}
            MATCH (s:Entity)-[:SUBJECT_OF]->(r)-[:OBJECT_OF]->(o:Entity)
            OPTIONAL MATCH (ch:Chunk)-[:ASSERTS]->(r)
            WITH r, s, o, score,
                 collect(DISTINCT {{chunk_id: ch.ChunkID, text: ch.text, file_name:ch.fileName}})[..3] AS sources
            RETURN r.relationId AS relation_id, s.name AS subject, o.name AS object,
                   r.relationshipLabel AS label, r.relationshipDescription AS description,
                   score AS vector_score, sources
            ORDER BY vector_score DESC
            LIMIT $top_k
        """,
            {"query_k": query_k, "top_k": top_k, "qvec": qvec, "domains": domains or None},
        ).data()


def _relation_window_chunks(rel_ids: list[str], max_chunks: int = 14) -> list[dict[str, Any]]:
    """Chunks asserting the given relations, ranked by support, widened prev/next via NEXT."""
    if not rel_ids:
        return []
    with neo4j_driver().session() as sess:
        return sess.run(
            """
            UNWIND $rel_ids AS rid
            MATCH (ch:Chunk)-[:ASSERTS]->(:Relation {relationId: rid})
            WITH ch,
                 collect(DISTINCT rid) AS matched_rel_ids,
                 count(DISTINCT rid) AS rel_support
            ORDER BY rel_support DESC
            LIMIT $maxc

            OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(ch)
            OPTIONAL MATCH (ch)-[:NEXT]->(nxt:Chunk)

            RETURN ch.ChunkID AS chunk_id,
                   ch.text AS text,
                   ch.fileName AS file_name,
                   ch.pageNumber AS page_number,
                   rel_support,
                   matched_rel_ids,
                   prev.ChunkID AS prev_id,
                   prev.text AS prev_text,
                   nxt.ChunkID AS next_id,
                   nxt.text AS next_text
        """,
            {"rel_ids": rel_ids, "maxc": max_chunks},
        ).data()


def _attach_graph_weight(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give every relation a uniform graph weight so the rerank blend is fair."""
    ids = [r["relation_id"] for r in relations]
    if not ids:
        return relations
    with neo4j_driver().session() as sess:
        wmap = {
            w["relation_id"]: w["weight"]
            for w in sess.run(
                """
            UNWIND $ids AS rid
            MATCH (r:Relation {relationId: rid})
            RETURN rid AS relation_id,
                   coalesce(r.aggregatedWeight, r.weight, r.relationshipStrength/10.0, 0) AS weight
        """,
                {"ids": ids},
            ).data()
        }
    for r in relations:
        r["weight"] = wmap.get(r["relation_id"], 0.0)
    return relations
