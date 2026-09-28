"""Pickle checkpoint state: init/load/save, all parameterized by an explicit
path (no hardcoded globals) so callers decide which checkpoint file they mean.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger("graph_construction.pipeline.checkpoint")


def init_empty_state() -> dict[str, Any]:
    return {
        "all_entity_nodes": [],
        "all_relation_nodes": [],
        "all_chunk_payloads": [],
        "all_mentions": [],
        "all_assertions": [],
        "file_name_to_extractions": {},
        "seen_entity_keys": set(),
        "seen_relation_ids": set(),
        "seen_chunk_ids": set(),
        "seen_mention_keys": set(),
        "seen_assertion_keys": set(),
        "completed_chunk_ids": set(),
        "completed_files": set(),
        "failed_chunks": [],
        "failed_chunk_ids": set(),
        "stats": {
            "chunks_processed": 0,
            "chunks_fully_removed": 0,
            "llm_errors": 0,
            "failed_chunks": 0,
            "last_saved_at": None,
        },
    }


def load_state(path: Path = config.STATE_FILE) -> dict[str, Any]:
    if path.exists():
        with open(path, "rb") as f:
            state = pickle.load(f)
        logger.info("Loaded checkpoint: %s", path)
        return state
    logger.info("No checkpoint found at %s. Starting fresh.", path)
    return init_empty_state()


def save_state(
    state: dict[str, Any],
    path: Path = config.STATE_FILE,
    tmp_path: Path | None = None,
) -> None:
    """Atomic write: dump to a tmp file, then rename over the real one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_path or path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    tmp_path.replace(path)


def maybe_save_checkpoint(
    state: dict[str, Any],
    path: Path = config.STATE_FILE,
    *,
    force: bool = False,
    chunk_counter_since_save: int = 0,
    file_counter_since_save: int = 0,
) -> bool:
    should_save = (
        force
        or chunk_counter_since_save >= config.CHECKPOINT_EVERY_N_CHUNKS
        or file_counter_since_save >= config.CHECKPOINT_EVERY_N_FILES
    )
    if should_save:
        state["stats"]["last_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_state(state, path)
        return True
    return False
