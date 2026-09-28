"""Synthesizes a single canonical description per entity from all its
per-chunk descriptions, for entities seen enough times to be worth merging.

run_entity_description_merge() is the entry point -- wraps collect -> parallel
LLM merge -> apply-to-state as one callable, mutating state["all_entity_nodes"]
in place and returning a summary dict.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from . import config, prompts
from .clients import call_llm
from .text_utils import entity_key_from_name

logger = logging.getLogger("graph_construction.pipeline.entity_merge")

_MERGE_SYSTEM_PROMPT = (
    "You are a senior specialist in tendering, ITB analysis, EPC contracts, "
    "procurement documents, and knowledge graph curation. "
    "You output STRICT JSON only."
)


def llm_merge_entity_description(entity_name: str, descriptions: list[str]) -> str:
    description_list = "\n".join(f"- {d.strip()}" for d in descriptions if d and d.strip())
    prompt = prompts.ENTITY_DESCRIPTION_SUMMARY_PROMPT.format(
        entity_name=entity_name,
        description_list=description_list,
        max_length=config.MAX_DESCRIPTION_WORDS,
    )
    response = call_llm(system_prompt=_MERGE_SYSTEM_PROMPT, user_prompt=prompt)
    data = json.loads(response.choices[0].message.content or "{}")
    return (data.get("description") or "").strip()


def merge_one_entity_description(
    entity_key: str, entity_name: str, descriptions: list[str], occurrence_count: int
) -> dict[str, Any]:
    try:
        merged_description = llm_merge_entity_description(entity_name, descriptions)
        return {
            "entity_key": entity_key,
            "entity_name": entity_name,
            "occurrence_count": occurrence_count,
            "description_count": len(descriptions),
            "updated": bool(merged_description),
            "description": merged_description,
            "error": None,
        }
    except Exception as e:
        return {
            "entity_key": entity_key,
            "entity_name": entity_name,
            "occurrence_count": occurrence_count,
            "description_count": len(descriptions),
            "updated": False,
            "description": "",
            "error": str(e),
        }


def _collect_description_jobs(state: dict, threshold: int) -> list[dict[str, Any]]:
    entity_descriptions: dict[str, list[str]] = defaultdict(list)
    entity_occurrences: Counter = Counter()
    entity_names: dict[str, Counter] = defaultdict(Counter)

    for file_extractions in state["file_name_to_extractions"].values():
        for item in file_extractions:
            for _chunk_id, extraction in item.items():
                for ent in extraction.get("entities", []) or []:
                    name = (ent.get("name") or "").strip()
                    if not name:
                        continue
                    entity_key = entity_key_from_name(name)
                    description = (ent.get("description") or "").strip()
                    entity_occurrences[entity_key] += 1
                    entity_names[entity_key][name] += 1
                    if description:
                        entity_descriptions[entity_key].append(description)

    logger.info("entity keys with occurrences: %d", len(entity_occurrences))
    logger.info("entity keys with descriptions: %d", len(entity_descriptions))

    jobs = []
    for entity_key, occurrence_count in entity_occurrences.items():
        if occurrence_count < 2 or occurrence_count > threshold:
            continue
        descriptions = entity_descriptions.get(entity_key, [])
        if not descriptions:
            continue
        entity_name = entity_names[entity_key].most_common(1)[0][0]
        jobs.append(
            {
                "entity_key": entity_key,
                "entity_name": entity_name,
                "descriptions": descriptions,
                "occurrence_count": occurrence_count,
            }
        )
    return jobs


def run_entity_description_merge(
    state: dict,
    max_workers: int = config.MAX_DESCRIPTION_MERGE_WORKERS,
    threshold: int = config.DESCRIPTION_MERGE_THRESHOLD,
) -> dict[str, Any]:
    """Collects per-chunk entity descriptions, merges entities seen 2..threshold
    times via parallel LLM calls, and applies the merged description back onto
    state["all_entity_nodes"] in place.
    """
    jobs = _collect_description_jobs(state, threshold)
    logger.info("description merge jobs: %d", len(jobs))

    merge_results: list[dict[str, Any]] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                merge_one_entity_description,
                job["entity_key"],
                job["entity_name"],
                job["descriptions"],
                job["occurrence_count"],
            ): job
            for job in jobs
        }
        for idx, future in enumerate(as_completed(future_to_job), start=1):
            result = future.result()
            merge_results.append(result)
            if result["error"]:
                logger.warning(
                    "[%d/%d] FAILED %s: %s", idx, len(jobs), result["entity_name"], result["error"]
                )
            else:
                logger.info("[%d/%d] merged %s", idx, len(jobs), result["entity_name"])

    logger.info("description merge LLM phase complete in %.1fs", time.time() - start)

    description_by_entity_key = {
        r["entity_key"]: r["description"]
        for r in merge_results
        if r.get("updated") and r.get("description")
    }
    updated_count = 0
    for ent in state["all_entity_nodes"]:
        entity_key = ent.get("entity_key")
        if entity_key in description_by_entity_key:
            ent["description"] = description_by_entity_key[entity_key]
            updated_count += 1

    success_count = sum(1 for r in merge_results if r.get("updated"))
    failed_count = sum(1 for r in merge_results if r.get("error"))

    state["entity_description_merge_log"] = merge_results
    state["stats"]["entity_descriptions_merged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["stats"]["entity_descriptions_updated"] = updated_count
    state["stats"]["entity_descriptions_success"] = success_count
    state["stats"]["entity_descriptions_failed"] = failed_count
    state["stats"]["entity_description_merge_threshold"] = threshold

    summary = {
        "jobs": len(jobs),
        "success_count": success_count,
        "failed_count": failed_count,
        "updated_count": updated_count,
    }
    logger.info("Description merge complete: %s", summary)
    return summary
