#!/usr/bin/env python3
"""CLI entrypoint for the semantic KG extraction pipeline -- the only module
in this package with side effects at run time. Every other module is a plain,
side-effect-free-on-import set of functions this file calls in sequence.

    cd graph_construction
    python -m pipeline.run_pipeline
    python -m pipeline.run_pipeline --retry-failed
    python -m pipeline.run_pipeline --skip-neo4j-write --skip-embeddings   # dry run
    python -m pipeline.run_pipeline --backfill-keywords                    # see relation_keywords.py

Stages: fetch chunks from Azure Search -> parallel extraction (or retry-failed)
-> entity description merge -> checkpoint snapshot -> Neo4j write (constraints,
entities, relations incl. keywords, mentions, assertions, aggregate refresh,
entity descriptions) -> optional keyword backfill -> relation embeddings +
vector index.
"""

from __future__ import annotations

import argparse
import logging

from . import checkpoint, config, entity_merge, extraction, neo4j_writer, relation_embeddings, relation_keywords, search_ingest
from .clients import get_neo4j_driver, get_search_client

logger = logging.getLogger("graph_construction.pipeline.run_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retry-failed", action="store_true",
                        help="only re-run chunks recorded in state['failed_chunks'], instead of a full extraction pass")
    parser.add_argument("--skip-neo4j-write", action="store_true", help="skip the Neo4j write stage entirely")
    parser.add_argument("--skip-embeddings", action="store_true", help="skip relation embedding + vector index creation")
    parser.add_argument("--backfill-keywords", action="store_true",
                        help="run the post-hoc relationship_keywords backfill for relations missing them "
                             "(normally unnecessary -- extraction now produces keywords directly)")
    parser.add_argument("--target-itb", default=config.TARGET_ITB_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("TARGET_ITB_ID=%s", args.target_itb)

    # ── Ingest: fetch chunks + resolve docURLs ────────────────────────────────
    search_client = get_search_client()
    all_files_docs = search_ingest.fetch_all_files_docs(search_client)
    file_name_url_map = search_ingest.build_file_name_url_map(get_neo4j_driver(), list(all_files_docs.keys()))

    # ── Extraction ─────────────────────────────────────────────────────────────
    state = checkpoint.load_state()
    try:
        if args.retry_failed:
            extraction.retry_failed_chunks(state, all_files_docs, file_name_url_map)
        else:
            extraction.run_parallel_extraction(all_files_docs, state, file_name_url_map)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving checkpoint...")
        checkpoint.save_state(state)
        raise

    logger.info(
        "Extraction state: entities=%d relations=%d chunks=%d mentions=%d assertions=%d failed=%d",
        len(state["all_entity_nodes"]), len(state["all_relation_nodes"]), len(state["all_chunk_payloads"]),
        len(state["all_mentions"]), len(state["all_assertions"]), len(state.get("failed_chunks", [])),
    )

    # ── Entity description merge ──────────────────────────────────────────────
    entity_merge.run_entity_description_merge(state)
    checkpoint.save_state(state, config.STATE_POST_MERGE_FILE)
    logger.info("Post-merge checkpoint saved -> %s", config.STATE_POST_MERGE_FILE)

    if args.skip_neo4j_write:
        logger.info("--skip-neo4j-write set. Skipping Neo4j write and everything after it.")
        return

    # ── Neo4j write ────────────────────────────────────────────────────────────
    driver = get_neo4j_driver()
    neo4j_writer.ensure_constraints(driver)
    neo4j_writer.write_entities(driver, state["all_entity_nodes"])
    neo4j_writer.write_relations(driver, state["all_relation_nodes"])
    neo4j_writer.write_mentions(driver, state["all_mentions"])
    neo4j_writer.write_assertions(driver, state["all_assertions"])
    neo4j_writer.refresh_relation_aggregates(driver)

    entity_description_rows = [
        {"entity_key": ent["entity_key"], "description": (ent.get("description") or "").strip()}
        for ent in state["all_entity_nodes"]
        if ent.get("entity_key") and (ent.get("description") or "").strip()
    ]
    neo4j_writer.write_entity_descriptions(driver, entity_description_rows)

    logger.info("Neo4j write complete: %s", neo4j_writer.get_graph_summary(driver))

    # ── Optional: backfill keywords for relations that still have none ───────
    if args.backfill_keywords:
        relation_keywords.run_high_level_keyword_backfill(driver)

    # ── Relation embeddings + vector index ────────────────────────────────────
    if not args.skip_embeddings:
        relation_embeddings.run_relation_embedding_pipeline(driver)
    else:
        logger.info("--skip-embeddings set. Done.")


if __name__ == "__main__":
    main()
