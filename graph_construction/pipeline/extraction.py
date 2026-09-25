"""Core extraction engine: one LLM call per chunk -> entities/relations,
merged thread-safely into a shared state dict, driven in parallel batches
across files.

Threading model matches the original notebook: process_single_chunk (the LLM
call) runs in worker threads with no shared-state access; merge_result_into_state
is the only place that touches `state`, guarded by _STATE_LOCK.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import APIConnectionError, APITimeoutError

from . import checkpoint, config, prompts
from .clients import call_llm
from .text_utils import normalize_keywords, resolve_section_hint

logger = logging.getLogger("graph_construction.pipeline.extraction")

_STATE_LOCK = threading.Lock()


def fallback_entity_description(raw: dict, cleaned_chunk: str = "") -> str:
    """Ensure every extracted entity has a grounded, non-empty description.

    If the LLM provided a description, keep it. If not, use a minimal
    non-hallucinated fallback based on category/type.
    """
    description = (raw.get("description") or "").strip()
    if description:
        return description

    category = ((raw.get("canonical_category") or "entity").strip() or "entity").replace("_", " ")
    llm_category = ((raw.get("llm_category") or "").strip()).replace("_", " ")

    if llm_category and llm_category != category:
        return f"Listed as a {category} in the source chunk, with source-specific type '{llm_category}'."
    return f"Listed as a {category} in the source chunk."


def process_single_chunk(
    chunk: dict,
    file_name: str,
    doc_url: Optional[str],
    extraction_prompt: str,
    active_entity_categories: Set[str],
    completed_chunk_ids_snapshot: Set[str],
) -> Optional[dict]:
    """Process one chunk independently. Returns a result dict with all
    extracted data, without touching global state -- the caller merges
    results under a lock (merge_result_into_state).
    """
    chunk_id = chunk["chunk_id"]

    if chunk_id in completed_chunk_ids_snapshot:
        return None

    chunk_text = (chunk.get("page_chunk", "") or "").strip()
    if len(chunk_text) < 40:
        return {"chunk_id": chunk_id, "skipped_short": True, "file_name": file_name}

    section_hint = resolve_section_hint(chunk_text)

    chunk_payload = {
        "chunk_id": chunk_id,
        "doc_url": doc_url,
        "sequence_number": chunk.get("sequence_number", 0),
        "page_number": chunk.get("page_number"),
        "text": chunk_text,
        "section_hint": section_hint,
        "file_name": file_name,
    }

    # ---- LLM call (the expensive part -- this is what we're parallelizing) ----
    try:
        response = call_llm(
            system_prompt=(
                "You are a Senior Legal Counsel and Procurement Specialist with deep expertise in "
                "international contract law and tender documentation. Your goal is to accurately "
                "extract knowledge graph entities and free-form relationship statements from complex "
                "tender clauses. You prioritize precision, adhere strictly to legal and technical "
                "terminology, and output your findings exclusively in valid, machine-readable JSON."
            ),
            user_prompt=(extraction_prompt + f"\n\nDOCUMENT: {file_name}\n\nCLAUSE TEXT:\n" + chunk_text[:9000]),
        )

        if response.choices[0].finish_reason == "length":
            logger.warning("truncated %s", chunk_id)

        extraction = json.loads(response.choices[0].message.content)

    except (APITimeoutError, APIConnectionError) as e:
        logger.warning("TIMEOUT/CONNECTION on %s: %s", chunk_id, e)
        return {
            "chunk_id": chunk_id, "error": True, "error_type": type(e).__name__, "error_msg": str(e),
            "file_name": file_name, "page_number": chunk.get("page_number"), "text_preview": chunk_text[:500],
        }
    except Exception as e:
        logger.warning("ERROR on %s: %s", chunk_id, e)
        return {
            "chunk_id": chunk_id, "error": True, "error_type": type(e).__name__, "error_msg": str(e),
            "file_name": file_name, "page_number": chunk.get("page_number"), "text_preview": chunk_text[:500],
        }

    result: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "file_name": file_name,
        "chunk_payload": chunk_payload,
        "extraction": extraction,
        "entity_nodes": [],
        "relation_nodes": [],
        "mentions": [],
        "assertions": [],
        "entity_map": {},
        "fully_removed": False,
    }

    if extraction.get("cleaning_status") == "fully_removed":
        result["fully_removed"] = True
        return result

    entity_map: Dict[str, str] = {}
    entity_description_by_name: Dict[str, str] = {}

    # --- Entities ---
    for raw in extraction.get("entities", []) or []:
        name = (raw.get("name") or "").strip()
        if not name:
            continue

        aliases = [a.strip() for a in (raw.get("aliases") or []) if isinstance(a, str) and a.strip()]
        normalized_name = re.sub(r"[^a-z0-9 ]+", " ", name.lower()).strip()

        canonical_category = re.sub(
            r"[^a-z0-9 ]+", " ", (raw.get("canonical_category") or "").lower()
        ).strip().replace(" ", "_")
        llm_category = re.sub(
            r"[^a-z0-9 ]+", " ", (raw.get("llm_category") or "").lower()
        ).strip().replace(" ", "_")

        if not canonical_category or canonical_category not in active_entity_categories:
            canonical_category = "other"
        if not llm_category:
            llm_category = canonical_category

        description = fallback_entity_description(
            {**raw, "canonical_category": canonical_category, "llm_category": llm_category},
            extraction.get("cleaned_chunk") or chunk_text,
        )

        entity_key = hashlib.sha1(f"entity||{normalized_name}".encode("utf-8")).hexdigest()
        entity_map[name] = entity_key
        entity_description_by_name[name] = description

        result["entity_nodes"].append({
            "entity_key": entity_key, "name": name, "normalized_name": normalized_name,
            "canonical_category": canonical_category, "llm_category": llm_category,
            "description": description, "aliases": aliases,
        })
        result["mentions"].append({"chunk_id": chunk_id, "entity_key": entity_key})

    # --- Relations ---
    for relation in extraction.get("relations", []) or []:
        subject_name = (relation.get("subject") or "").strip()
        object_name = (relation.get("object") or "").strip()

        relationship_label = re.sub(
            r"[^a-z0-9 ]+", " ", (relation.get("relationship_label") or "").lower()
        ).strip().replace(" ", "_")
        relationship_description = (relation.get("relationship_description") or "").strip()
        evidence_text = (relation.get("evidence") or "").strip()

        if not subject_name or not object_name:
            continue
        if not relationship_label or not relationship_description:
            continue

        subject_norm = re.sub(r"[^a-z0-9 ]+", " ", subject_name.lower()).strip()
        object_norm = re.sub(r"[^a-z0-9 ]+", " ", object_name.lower()).strip()

        subject_canonical_category = re.sub(
            r"[^a-z0-9 ]+", " ", (relation.get("subject_canonical_category") or "").lower()
        ).strip().replace(" ", "_")
        object_canonical_category = re.sub(
            r"[^a-z0-9 ]+", " ", (relation.get("object_canonical_category") or "").lower()
        ).strip().replace(" ", "_")
        subject_llm_category = re.sub(
            r"[^a-z0-9 ]+", " ", (relation.get("subject_llm_category") or "").lower()
        ).strip().replace(" ", "_")
        object_llm_category = re.sub(
            r"[^a-z0-9 ]+", " ", (relation.get("object_llm_category") or "").lower()
        ).strip().replace(" ", "_")

        subject_category = subject_canonical_category if subject_canonical_category in active_entity_categories else "other"
        object_category = object_canonical_category if object_canonical_category in active_entity_categories else "other"

        if not subject_llm_category:
            subject_llm_category = subject_category
        if not object_llm_category:
            object_llm_category = object_category

        subject_key = entity_map.get(subject_name) or hashlib.sha1(f"entity||{subject_norm}".encode("utf-8")).hexdigest()
        object_key = entity_map.get(object_name) or hashlib.sha1(f"entity||{object_norm}".encode("utf-8")).hexdigest()

        subject_description = entity_description_by_name.get(subject_name) or fallback_entity_description(
            {"name": subject_name, "canonical_category": subject_category,
             "llm_category": subject_llm_category, "description": ""},
            extraction.get("cleaned_chunk") or chunk_text,
        )
        object_description = entity_description_by_name.get(object_name) or fallback_entity_description(
            {"name": object_name, "canonical_category": object_category,
             "llm_category": object_llm_category, "description": ""},
            extraction.get("cleaned_chunk") or chunk_text,
        )

        result["entity_nodes"].append({
            "entity_key": subject_key, "name": subject_name, "normalized_name": subject_norm,
            "canonical_category": subject_category, "llm_category": subject_llm_category,
            "description": subject_description, "aliases": [],
        })
        result["mentions"].append({"chunk_id": chunk_id, "entity_key": subject_key})

        result["entity_nodes"].append({
            "entity_key": object_key, "name": object_name, "normalized_name": object_norm,
            "canonical_category": object_category, "llm_category": object_llm_category,
            "description": object_description, "aliases": [],
        })
        result["mentions"].append({"chunk_id": chunk_id, "entity_key": object_key})

        try:
            confidence = float(relation.get("confidence", 0.75))
        except (TypeError, ValueError):
            confidence = 0.75
        confidence = max(0.0, min(confidence, 1.0))

        try:
            relationship_strength = int(relation.get("relationship_strength", 5))
        except (TypeError, ValueError):
            relationship_strength = 5
        relationship_strength = max(1, min(relationship_strength, 10))

        # Extraction-time high-level theme tags (see prompts.build_semantic_extraction_prompt's
        # relationship_keywords rule) -- normalized the same way the post-hoc backfill pass
        # (relation_keywords.py) normalizes them, so the two are interchangeable downstream.
        relationship_keywords = normalize_keywords(
            relation.get("relationship_keywords"), label=relationship_label,
        )

        if not evidence_text:
            evidence_text = chunk_payload["text"][:1000].strip()

        relation_id = hashlib.sha1(
            f"relation||{subject_key}||{relationship_label}||{object_key}||{relationship_description.lower()}".encode("utf-8")
        ).hexdigest()

        result["relation_nodes"].append({
            "relation_id": relation_id,
            "relationship_label": relationship_label,
            "relationship_description": relationship_description,
            "relationship_strength": relationship_strength,
            "relationship_keywords": relationship_keywords,
            "subject_key": subject_key,
            "object_key": object_key,
        })

        assertion_key = hashlib.sha1(f"assertion||{chunk_id}||{relation_id}".encode("utf-8")).hexdigest()
        result["assertions"].append({
            "assertion_key": assertion_key, "chunk_id": chunk_id, "relation_id": relation_id,
            "evidence_text": evidence_text, "confidence": confidence,
            "relationship_strength": relationship_strength,
            "page_number": chunk_payload["page_number"], "doc_url": chunk_payload["doc_url"],
            "file_name": file_name,
        })

    result["entity_map"] = entity_map
    return result


def merge_result_into_state(result: Optional[dict], state: dict,
                            failed_chunks: Optional[Dict[str, list]] = None) -> None:
    """Thread-safe merge of one chunk's results into global state."""
    if result is None:
        return

    chunk_id = result["chunk_id"]
    file_name = result["file_name"]

    with _STATE_LOCK:
        if chunk_id in state["completed_chunk_ids"]:
            return

        if result.get("error"):
            failed_record = {
                "chunk_id": chunk_id, "file_name": file_name, "page_number": result.get("page_number"),
                "error_type": result.get("error_type"), "error": result.get("error_msg"),
                "text_preview": result.get("text_preview", ""),
            }
            state.setdefault("failed_chunks", [])
            state.setdefault("failed_chunk_ids", set())
            if chunk_id not in state["failed_chunk_ids"]:
                state["failed_chunk_ids"].add(chunk_id)
                state["failed_chunks"].append(failed_record)
            if failed_chunks is not None:
                failed_chunks[file_name].append({
                    "chunk_id": chunk_id, "error_type": result.get("error_type"), "error": result.get("error_msg"),
                })
            state["stats"]["llm_errors"] += 1
            state["stats"]["failed_chunks"] = len(state["failed_chunks"])
            return

        if result.get("skipped_short"):
            state["completed_chunk_ids"].add(chunk_id)
            state["stats"]["chunks_processed"] += 1
            return

        if result.get("fully_removed"):
            state["completed_chunk_ids"].add(chunk_id)
            state["stats"]["chunks_processed"] += 1
            state["stats"]["chunks_fully_removed"] += 1
            return

        if chunk_id not in state["seen_chunk_ids"]:
            state["all_chunk_payloads"].append(result["chunk_payload"])
            state["seen_chunk_ids"].add(chunk_id)

        for ent in result["entity_nodes"]:
            if ent["entity_key"] not in state["seen_entity_keys"]:
                state["all_entity_nodes"].append(ent)
                state["seen_entity_keys"].add(ent["entity_key"])

        for m in result["mentions"]:
            mention_key = f"{m['chunk_id']}||{m['entity_key']}"
            if mention_key not in state["seen_mention_keys"]:
                state["all_mentions"].append(m)
                state["seen_mention_keys"].add(mention_key)

        for rel in result["relation_nodes"]:
            if rel["relation_id"] not in state["seen_relation_ids"]:
                state["all_relation_nodes"].append(rel)
                state["seen_relation_ids"].add(rel["relation_id"])

        for a in result["assertions"]:
            if a["assertion_key"] not in state["seen_assertion_keys"]:
                state["all_assertions"].append(a)
                state["seen_assertion_keys"].add(a["assertion_key"])

        state["completed_chunk_ids"].add(chunk_id)
        state["stats"]["chunks_processed"] += 1


def process_single_file(
    file_name: str,
    doc_chunks: List[dict],
    state_snapshot: dict,
    active_entity_categories: Set[str],
    extraction_prompt: str,
    file_name_url_map: Dict[str, Optional[str]],
) -> Tuple[Optional[List[dict]], Optional[List[dict]], str]:
    """Process all chunks of one file, sequentially within the file (the LLM
    calls happen here, inside the worker thread)."""
    doc_url = file_name_url_map.get(file_name)
    if not doc_url:
        logger.warning("SKIP (no docURL): %s", file_name)
        return None, None, file_name

    logger.info("[thread] Processing: %s (%d chunks)", file_name.split("/")[-1], len(doc_chunks))

    completed_snapshot = state_snapshot["completed_chunk_ids"]
    chunk_results = []
    file_extractions = []

    for chunk in doc_chunks:
        result = process_single_chunk(
            chunk=chunk, file_name=file_name, doc_url=doc_url,
            extraction_prompt=extraction_prompt, active_entity_categories=active_entity_categories,
            completed_chunk_ids_snapshot=completed_snapshot,
        )
        chunk_results.append(result)

        if result and not result.get("error") and not result.get("skipped_short"):
            extraction = result.get("extraction")
            if extraction:
                file_extractions.append({result["chunk_id"]: extraction})

    return chunk_results, file_extractions, file_name


def _category_distribution(items: List[dict], key: str) -> Dict[Any, int]:
    return dict(Counter(it.get(key) for it in items if it.get(key)).most_common())


def run_parallel_extraction(
    file_name_chunks_included: Dict[str, List[dict]],
    state: dict,
    file_name_url_map: Dict[str, Optional[str]],
    failed_chunks: Optional[Dict[str, list]] = None,
    checkpoint_path: Path = config.STATE_FILE,
) -> None:
    """Divides files into batches of config.FILE_BATCH_SIZE and processes each
    batch concurrently. Writes a checkpoint + a per-batch JSON log after every
    batch. Entity ontology is fixed, loaded once from entity_category_cards.json;
    relations are free-form (relationship_label/description/strength/keywords).
    """
    if failed_chunks is None:
        failed_chunks = defaultdict(list)

    state.setdefault("failed_chunks", [])
    state.setdefault("failed_chunk_ids", set())
    state["stats"].setdefault("failed_chunks", len(state["failed_chunks"]))

    all_entity_nodes = state["all_entity_nodes"]
    all_relation_nodes = state["all_relation_nodes"]
    all_mentions = state["all_mentions"]
    all_assertions = state["all_assertions"]
    file_name_to_extractions = state["file_name_to_extractions"]
    completed_files = state["completed_files"]

    dynamic_entity_category_block = prompts.render_entity_category_block(prompts.load_entity_cards())
    extraction_prompt = prompts.build_semantic_extraction_prompt(dynamic_entity_category_block)
    active_entity_categories = set(prompts.canonical_entity_categories())

    logger.info("Fixed ontology: %d entity categories", len(active_entity_categories))
    logger.info("Prompt size: %s chars", f"{len(extraction_prompt):,}")

    pending_files = [
        (fname, chunks) for fname, chunks in file_name_chunks_included.items()
        if fname not in completed_files
    ]
    total_files = len(pending_files)
    logger.info("Total files to process: %d (batch size: %d)", total_files, config.FILE_BATCH_SIZE)

    batches = [pending_files[i:i + config.FILE_BATCH_SIZE] for i in range(0, total_files, config.FILE_BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        logger.info("BATCH %d/%d -- %d files", batch_idx + 1, len(batches), len(batch))

        with _STATE_LOCK:
            counts_before = {
                "entity_nodes": len(all_entity_nodes), "relation_nodes": len(all_relation_nodes),
                "mentions": len(all_mentions), "assertions": len(all_assertions),
                "failed_chunks": len(state["failed_chunks"]),
            }
            state_snapshot = {"completed_chunk_ids": set(state["completed_chunk_ids"])}

        batch_start_ts = time.time()
        batch_file_stats = []
        batch_errors = []

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_PER_BATCH) as executor:
            future_to_file = {
                executor.submit(
                    process_single_file, file_name=fname, doc_chunks=chunks,
                    state_snapshot=state_snapshot, active_entity_categories=active_entity_categories,
                    extraction_prompt=extraction_prompt, file_name_url_map=file_name_url_map,
                ): fname
                for fname, chunks in batch
            }

            for future in as_completed(future_to_file):
                fname = future_to_file[future]
                try:
                    chunk_results, file_extractions, file_name = future.result()
                except Exception as e:
                    logger.error("FATAL error processing file %s: %s", fname, e)
                    batch_errors.append({"file": fname, "error": str(e)})
                    continue

                if chunk_results is None:
                    with _STATE_LOCK:
                        completed_files.add(file_name)
                    batch_file_stats.append({
                        "file": file_name, "status": "no_doc_url", "n_chunks": 0,
                        "n_entities": 0, "n_relations": 0, "n_failed_chunks": 0,
                    })
                    continue

                for result in chunk_results:
                    merge_result_into_state(result, state, failed_chunks)

                valid_results = [r for r in chunk_results if r and not r.get("error") and not r.get("skipped_short")]
                n_ents = sum(len((r.get("extraction") or {}).get("entities", []) or []) for r in valid_results)
                n_rels = sum(len((r.get("extraction") or {}).get("relations", []) or []) for r in valid_results)
                n_failed = sum(1 for r in chunk_results if r and r.get("error"))

                with _STATE_LOCK:
                    file_name_to_extractions[file_name] = file_extractions or []
                    completed_files.add(file_name)

                batch_file_stats.append({
                    "file": file_name, "status": "ok",
                    "n_chunks": len([r for r in chunk_results if r is not None]),
                    "n_entities": n_ents, "n_relations": n_rels, "n_failed_chunks": n_failed,
                })
                logger.info(
                    "done -- %s | entity_nodes: %d | relation_nodes: %d | mentions: %d | assertions: %d | failed_chunks: %d",
                    file_name.split("/")[-1], len(all_entity_nodes), len(all_relation_nodes),
                    len(all_mentions), len(all_assertions), len(state["failed_chunks"]),
                )

        batch_elapsed_s = round(time.time() - batch_start_ts, 1)

        with _STATE_LOCK:
            counts_after = {
                "entity_nodes": len(all_entity_nodes), "relation_nodes": len(all_relation_nodes),
                "mentions": len(all_mentions), "assertions": len(all_assertions),
                "failed_chunks": len(state["failed_chunks"]),
            }
            entity_dist = _category_distribution(all_entity_nodes, "canonical_category")
            relation_label_dist = _category_distribution(all_relation_nodes, "relationship_label")
            relation_strength_dist = _category_distribution(all_relation_nodes, "relationship_strength")

        delta = {k: counts_after[k] - counts_before[k] for k in counts_before}

        with _STATE_LOCK:
            state["stats"]["last_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            state["stats"]["failed_chunks"] = len(state["failed_chunks"])
            checkpoint.save_state(state, checkpoint_path)

        batch_log = {
            "batch_idx": batch_idx, "batch_label": f"{batch_idx + 1}/{len(batches)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_seconds": batch_elapsed_s,
            "n_files_in_batch": len(batch),
            "n_files_ok": sum(1 for f in batch_file_stats if f["status"] == "ok"),
            "n_files_skipped": sum(1 for f in batch_file_stats if f["status"] == "no_doc_url"),
            "n_files_errored": len(batch_errors), "errors": batch_errors,
            "counts_before": counts_before, "counts_after": counts_after, "delta": delta,
            "files": batch_file_stats,
            "cumulative_entity_distribution": entity_dist,
            "cumulative_relation_label_distribution": relation_label_dist,
            "cumulative_relation_strength_distribution": relation_strength_dist,
            "completed_files_total": len(completed_files),
            "completed_chunks_total": len(state["completed_chunk_ids"]),
            "failed_chunks_total": len(state["failed_chunks"]),
        }
        log_path = config.BATCH_LOG_DIR / f"batch_{batch_idx:04d}.json"
        log_path.write_text(json.dumps(batch_log, indent=2, ensure_ascii=False))

        logger.info(
            "Checkpoint saved | batch took %ss | +%d entities, +%d relations, +%d failed chunks | log: %s",
            batch_elapsed_s, delta["entity_nodes"], delta["relation_nodes"], delta["failed_chunks"], log_path.name,
        )


def retry_failed_chunks(
    state: dict,
    file_name_chunks_included: Dict[str, List[dict]],
    file_name_url_map: Dict[str, Optional[str]],
    checkpoint_path: Path = config.STATE_FILE,
) -> Dict[str, int]:
    """Re-extract only the chunks listed in state["failed_chunks"] and merge
    successful results back into state. Real, callable version of what used
    to be a commented-out manual-retry notebook cell.
    """
    dynamic_entity_category_block = prompts.render_entity_category_block(prompts.load_entity_cards())
    extraction_prompt = prompts.build_semantic_extraction_prompt(dynamic_entity_category_block)
    active_entity_categories = set(prompts.canonical_entity_categories())

    failed_records = list(state.get("failed_chunks", []))
    failed_chunk_ids = {rec.get("chunk_id") for rec in failed_records if rec.get("chunk_id")}
    logger.info("Failed chunks to retry: %d", len(failed_chunk_ids))

    chunk_lookup: Dict[str, dict] = {}
    for file_name, chunks in file_name_chunks_included.items():
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id in failed_chunk_ids:
                chunk_lookup[chunk_id] = {"chunk": chunk, "file_name": file_name}

    missing = failed_chunk_ids - set(chunk_lookup)
    if missing:
        logger.warning("Missing chunk IDs not found in file_name_chunks_included: %s", sorted(missing))
    logger.info("Located failed chunks: %d / %d", len(chunk_lookup), len(failed_chunk_ids))

    # Clear failed-tracking for the chunks we're about to retry -- successful
    # retries stay removed; failures will be re-added by merge_result_into_state.
    with _STATE_LOCK:
        state["failed_chunks"] = [
            rec for rec in state.get("failed_chunks", []) if rec.get("chunk_id") not in failed_chunk_ids
        ]
        state["failed_chunk_ids"] = {cid for cid in state.get("failed_chunk_ids", set()) if cid not in failed_chunk_ids}
        state["completed_chunk_ids"] = {cid for cid in state["completed_chunk_ids"] if cid not in failed_chunk_ids}
        state["stats"]["failed_chunks"] = len(state["failed_chunks"])

    retry_results = []
    retry_failed_tracking: Dict[str, list] = defaultdict(list)

    for chunk_id, payload in chunk_lookup.items():
        chunk = payload["chunk"]
        file_name = payload["file_name"]
        doc_url = file_name_url_map.get(file_name)

        if not doc_url:
            result = {
                "chunk_id": chunk_id, "file_name": file_name, "page_number": chunk.get("page_number"),
                "error": True, "error_type": "MissingDocURL",
                "error_msg": f"Could not resolve doc_url for file: {file_name}",
                "text_preview": (chunk.get("page_chunk") or "")[:500],
            }
        else:
            logger.info("Retrying chunk: %s (file=%s, page=%s)", chunk_id, file_name, chunk.get("page_number"))
            result = process_single_chunk(
                chunk=chunk, file_name=file_name, doc_url=doc_url,
                extraction_prompt=extraction_prompt, active_entity_categories=active_entity_categories,
                completed_chunk_ids_snapshot=set(),
            )
        retry_results.append(result)

    for result in retry_results:
        merge_result_into_state(result, state, retry_failed_tracking)

    with _STATE_LOCK:
        for result in retry_results:
            if not result or result.get("error") or result.get("skipped_short"):
                continue
            extraction = result.get("extraction")
            if not extraction:
                continue
            file_name = result["file_name"]
            chunk_id = result["chunk_id"]
            existing = state["file_name_to_extractions"].setdefault(file_name, [])
            cleaned_existing = [item for item in existing if not (isinstance(item, dict) and chunk_id in item)]
            cleaned_existing.append({chunk_id: extraction})
            state["file_name_to_extractions"][file_name] = cleaned_existing

        state["stats"]["failed_chunks"] = len(state.get("failed_chunks", []))
        state["stats"]["last_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        checkpoint.save_state(state, checkpoint_path)

    summary = {
        "retried": len(retry_results),
        "succeeded": sum(1 for r in retry_results if r and not r.get("error") and not r.get("skipped_short")),
        "skipped_short": sum(1 for r in retry_results if r and r.get("skipped_short")),
        "still_failed": sum(1 for r in retry_results if r and r.get("error")),
    }
    logger.info("Retry complete: %s", summary)
    return summary
