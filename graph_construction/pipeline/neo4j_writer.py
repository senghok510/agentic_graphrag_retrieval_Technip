"""All Neo4j writes for the semantic KG layer: constraints/indexes, Entity
and Relation nodes, MENTIONS/ASSERTS edges, aggregate refresh, and the
entity-description write-back after merging.

Each function opens its own session off the given driver -- call them in
sequence from run_pipeline.py. Assumes ITB/Document/Chunk nodes already exist
from the earlier ingestion stage (this module only adds the semantic layer on
top of existing Chunk nodes, matched by ChunkID).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("graph_construction.pipeline.neo4j_writer")


def ensure_constraints(driver) -> None:
    statements = [
        "CREATE CONSTRAINT itb_id_unique IF NOT EXISTS FOR (i:ITB) REQUIRE i.ITB_ID IS UNIQUE",
        "CREATE CONSTRAINT document_url_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.docURL IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.ChunkID IS UNIQUE",
        "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entityKey IS UNIQUE",
        "CREATE CONSTRAINT relation_id_unique IF NOT EXISTS FOR (r:Relation) REQUIRE r.relationId IS UNIQUE",
        "CREATE INDEX entity_normalized_name IF NOT EXISTS FOR (e:Entity) ON (e.normalizedName)",
        "CREATE INDEX entity_canonical_category IF NOT EXISTS FOR (e:Entity) ON (e.canonicalCategory)",
        "CREATE INDEX relation_label IF NOT EXISTS FOR (r:Relation) ON (r.relationshipLabel)",
        "CREATE INDEX relation_strength IF NOT EXISTS FOR (r:Relation) ON (r.relationshipStrength)",
    ]
    with driver.session() as session:
        for stmt in statements:
            session.run(stmt)
    logger.info("Constraints/indexes ensured")


def write_entities(driver, entities: List[Dict[str, Any]]) -> None:
    with driver.session() as session:
        for entity in entities:
            session.run("""
                MERGE (e:Entity {entityKey: $entity_key})
                SET e.name = $name,
                    e.normalizedName = $normalized_name,
                    e.canonicalCategory = $canonical_category,
                    e.llmCategory = $llm_category,
                    e.description = $description,
                    e.aliases = $aliases
            """, {
                "entity_key": entity["entity_key"],
                "name": entity["name"],
                "normalized_name": entity["normalized_name"],
                "canonical_category": entity["canonical_category"],
                "llm_category": entity["llm_category"],
                "description": entity.get("description", ""),
                "aliases": entity.get("aliases", []),
            })
    logger.info("Wrote %d entity nodes", len(entities))


def write_relations(driver, relations: List[Dict[str, Any]]) -> None:
    """MERGEs each Relation node + SUBJECT_OF/OBJECT_OF edges, including
    relationshipKeywords straight from extraction (see extraction.py's
    process_single_chunk / prompts.build_semantic_extraction_prompt) -- this
    is the main path now; relation_keywords.py's post-hoc pass is only a
    backfill for relations that predate this or came back with no keywords."""
    with driver.session() as session:
        for rel in relations:
            session.run("""
                MATCH (subject:Entity {entityKey: $subject_key})
                MATCH (object:Entity {entityKey: $object_key})
                MERGE (r:Relation {relationId: $relation_id})
                SET r.relationshipLabel = $relationship_label,
                    r.relationshipDescription = $relationship_description,
                    r.relationshipStrength = $relationship_strength,
                    r.weight = toFloat($relationship_strength) / 10.0,
                    r.relationshipKeywords = $relationship_keywords,
                    r.subjectKey = $subject_key,
                    r.objectKey = $object_key
                MERGE (subject)-[:SUBJECT_OF]->(r)
                MERGE (r)-[:OBJECT_OF]->(object)
            """, {
                "relation_id": rel["relation_id"],
                "relationship_label": rel.get("relationship_label", ""),
                "relationship_description": rel.get("relationship_description", ""),
                "relationship_strength": int(rel.get("relationship_strength", 5)),
                "relationship_keywords": rel.get("relationship_keywords", []),
                "subject_key": rel["subject_key"],
                "object_key": rel["object_key"],
            })
    logger.info("Wrote %d relation nodes", len(relations))


def write_mentions(driver, mentions: List[Dict[str, Any]]) -> None:
    with driver.session() as session:
        for mention in mentions:
            session.run("""
                MATCH (ch:Chunk {ChunkID: $chunk_id})
                MATCH (e:Entity {entityKey: $entity_key})
                MERGE (ch)-[:MENTIONS]->(e)
            """, {"chunk_id": mention["chunk_id"], "entity_key": mention["entity_key"]})
    logger.info("Wrote %d MENTIONS edges", len(mentions))


def write_assertions(driver, assertions: List[Dict[str, Any]]) -> None:
    with driver.session() as session:
        for assertion in assertions:
            session.run("""
                MATCH (ch:Chunk {ChunkID: $chunk_id})
                MATCH (r:Relation {relationId: $relation_id})
                MERGE (ch)-[a:ASSERTS {assertionKey: $assertion_key}]->(r)
                SET a.evidenceText = $evidence_text,
                    a.confidence = $confidence,
                    a.relationshipStrength = $relationship_strength,
                    a.weight = toFloat($relationship_strength) / 10.0,
                    a.pageNumber = $page_number,
                    a.docURL = $doc_url,
                    a.fileName = $file_name
            """, {
                "assertion_key": assertion["assertion_key"],
                "chunk_id": assertion["chunk_id"],
                "relation_id": assertion["relation_id"],
                "evidence_text": assertion["evidence_text"],
                "confidence": float(assertion.get("confidence", 0.75)),
                "relationship_strength": int(assertion.get("relationship_strength", 5)),
                "page_number": assertion.get("page_number"),
                "doc_url": assertion.get("doc_url"),
                "file_name": assertion.get("file_name", ""),
            })
    logger.info("Wrote %d ASSERTS edges", len(assertions))


def refresh_relation_aggregates(driver) -> None:
    """Recomputes supportCount/avgConfidence/avgRelationshipStrength/
    aggregatedWeight on every Relation node from its ASSERTS edges."""
    with driver.session() as session:
        session.run("""
            MATCH (:Chunk)-[a:ASSERTS]->(r:Relation)
            WITH r,
                 count(a) AS support_count,
                 avg(coalesce(a.confidence, 0.75)) AS avg_confidence,
                 avg(coalesce(a.relationshipStrength, r.relationshipStrength, 5)) AS avg_strength
            SET r.supportCount = support_count,
                r.avgConfidence = avg_confidence,
                r.avgRelationshipStrength = avg_strength,
                r.aggregatedWeight = support_count * (avg_strength / 10.0) * avg_confidence
        """)
    logger.info("Refreshed relation aggregates")


def write_entity_descriptions(driver, entity_description_rows: List[Dict[str, str]]) -> int:
    """rows: [{"entity_key": ..., "description": ...}, ...] -- the merged
    descriptions from entity_merge.run_entity_description_merge()."""
    with driver.session() as session:
        result = session.run("""
            UNWIND $rows AS row
            MATCH (e:Entity {entityKey: row.entity_key})
            SET e.description = row.description
            RETURN count(e) AS updated_entities
        """, {"rows": entity_description_rows}).single()
    updated = result["updated_entities"]
    logger.info("Updated Neo4j entity descriptions: %d", updated)
    return updated


def get_graph_summary(driver) -> Dict[str, int]:
    with driver.session() as session:
        summary = session.run("""
            RETURN
                count { (i:ITB) }       AS itbs,
                count { (d:Document) }  AS documents,
                count { (ch:Chunk) }    AS chunks,
                count { (e:Entity) }    AS entities,
                count { (r:Relation) }  AS relations
        """).single()
    return dict(summary)
