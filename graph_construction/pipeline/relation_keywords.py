"""Post-hoc, per-relation high-level keyword extraction.

BACKFILL ONLY. Since prompts.build_semantic_extraction_prompt now asks for
relationship_keywords directly during extraction (and extraction.py /
neo4j_writer.py capture + persist it on the main path), this pass is only
useful for relations that predate that change, or that came back from
extraction with no keywords. run_high_level_keyword_backfill() only targets
relations where r.relationshipKeywords is missing/empty -- it will not
reprocess relations that already have keywords.

Fixes two bugs present in the original notebook cell, where the loop read
rel["relationID"] / rel["relationshipDescription"] (KeyErrors -- the Cypher
query actually returns relation_id / relationship_description) and never
fetched relationship_label at all (so the normalize_keywords() fallback label
was always None). As written, that cell could not have completed successfully.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from . import prompts
from .clients import call_llm
from .text_utils import normalize_keywords

logger = logging.getLogger("graph_construction.pipeline.relation_keywords")

MAX_KEYWORD_WORKERS = 8


def _fetch_relations_missing_keywords(driver, limit: int | None = None) -> list[dict[str, Any]]:
    query = """
        MATCH (r:Relation)
        WHERE r.relationshipKeywords IS NULL OR size(r.relationshipKeywords) = 0
        MATCH (d:Document)<-[:PART_OF]-(c:Chunk)-[:ASSERTS]->(r)
        RETURN DISTINCT
            r.relationId              AS relation_id,
            r.relationshipLabel       AS relationship_label,
            r.relationshipDescription AS relationship_description,
            c.text                    AS chunk_text
    """
    if limit:
        query += "\nLIMIT $limit"
    with driver.session() as session:
        return [row.data() for row in session.run(query, {"limit": limit} if limit else {})]


def _classify_one(rel: dict[str, Any]) -> dict[str, Any]:
    user_prompt = prompts.HIGH_LEVEL_PROMPT_USER.format(
        relationship_description=rel.get("relationship_description") or "",
        chunk=(rel.get("chunk_text") or "")[:2000],
    )
    try:
        resp = call_llm(prompts.HIGH_LEVEL_PROMPT_SYS, user_prompt)
        parsed = json.loads(resp.choices[0].message.content)
        keywords = normalize_keywords(
            parsed.get("relationship_keywords"), label=rel.get("relationship_label")
        )
    except Exception as e:
        logger.warning("keyword backfill failed for %s: %s", rel["relation_id"], e)
        keywords = normalize_keywords(None, label=rel.get("relationship_label"))
    return {"relation_id": rel["relation_id"], "keywords": keywords}


def run_high_level_keyword_backfill(
    driver,
    limit: int | None = None,
    max_workers: int = MAX_KEYWORD_WORKERS,
) -> dict[str, int]:
    relations = _fetch_relations_missing_keywords(driver, limit=limit)
    logger.info("Relations missing relationshipKeywords: %d", len(relations))
    if not relations:
        return {"processed": 0, "written": 0}

    keywords_by_relation: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_classify_one, rel): rel for rel in relations}
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            keywords_by_relation[result["relation_id"]] = result["keywords"]
            if idx % 50 == 0 or idx == len(relations):
                logger.info("keyword backfill: %d/%d", idx, len(relations))

    rows = [{"relationId": rid, "keywords": kws} for rid, kws in keywords_by_relation.items()]
    with driver.session() as session:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (r:Relation {relationId: row.relationId})
            SET r.relationshipKeywords = row.keywords
        """,
            {"rows": rows},
        )

    logger.info("Keyword backfill wrote %d relations", len(rows))
    return {"processed": len(relations), "written": len(rows)}
