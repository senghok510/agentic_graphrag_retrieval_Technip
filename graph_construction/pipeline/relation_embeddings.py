"""Embeds every Relation (LightRAG "global channel" text: keywords + endpoints
+ description) and writes the vectors + vector index Neo4j-native, so
chatbot/src/pipeline/graph_retrieval.py's _global_search can query them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from . import config
from .clients import embed_texts_large_model

logger = logging.getLogger("graph_construction.pipeline.relation_embeddings")


def build_relation_embed_text(row: Dict[str, Any]) -> str:
    kw = ", ".join(row.get("keywords") or [])
    return f"{kw}\t{row['subject']} {row['object']}\n{row['description']}"


def fetch_relation_rows(driver) -> List[Dict[str, Any]]:
    with driver.session() as session:
        return session.run("""
            MATCH (s:Entity)-[:SUBJECT_OF]->(r:Relation)-[:OBJECT_OF]->(o:Entity)
            RETURN r.relationId AS relation_id, s.name AS subject, o.name AS object,
                   r.relationshipDescription AS description, coalesce(r.relationshipKeywords, []) AS keywords
        """).data()


def embed_relations(rel_rows: List[Dict[str, Any]], batch_size: int = config.EMBED_BATCH) -> List[List[float]]:
    embed_texts = [build_relation_embed_text(r) for r in rel_rows]
    embeddings: List[List[float]] = []
    for i in range(0, len(embed_texts), batch_size):
        embeddings.extend(embed_texts_large_model(embed_texts[i:i + batch_size]))
        logger.info("embedded %d/%d", min(i + batch_size, len(embed_texts)), len(embed_texts))
    return embeddings


def write_relation_embeddings(driver, rel_rows: List[Dict[str, Any]], embeddings: List[List[float]],
                              batch_size: int = 1000) -> None:
    embed_rows = [{"relation_id": r["relation_id"], "embedding": emb} for r, emb in zip(rel_rows, embeddings)]
    with driver.session() as session:
        for i in range(0, len(embed_rows), batch_size):
            session.run("""
                UNWIND $rows AS row
                MATCH (r:Relation {relationId: row.relation_id})
                CALL db.create.setNodeVectorProperty(r, 'embedding', row.embedding)
            """, {"rows": embed_rows[i:i + batch_size]})
    logger.info("Wrote embeddings to %d relations", len(embed_rows))


def create_relation_vector_index(driver, dimensions: int) -> None:
    with driver.session() as session:
        session.run(f"""
            CREATE VECTOR INDEX relation_embedding_index IF NOT EXISTS
            FOR (r:Relation) ON (r.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }} }}
        """)
    logger.info("relation_embedding_index ensured (dimensions=%d)", dimensions)


def run_relation_embedding_pipeline(driver) -> Dict[str, int]:
    """Fetch -> embed -> write vectors -> create/ensure the vector index."""
    rel_rows = fetch_relation_rows(driver)
    logger.info("Relations to embed: %d", len(rel_rows))
    if not rel_rows:
        return {"embedded": 0}

    embeddings = embed_relations(rel_rows)
    dim = len(embeddings[0])
    logger.info("Embedded %d relations, dim=%d", len(embeddings), dim)

    write_relation_embeddings(driver, rel_rows, embeddings)
    create_relation_vector_index(driver, dim)
    return {"embedded": len(embeddings), "dimensions": dim}
