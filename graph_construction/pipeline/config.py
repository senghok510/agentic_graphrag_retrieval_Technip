"""Environment loading, logging setup, and every constant the pipeline uses.

Single source of truth -- the original notebook redefined CHECKPOINT_DIR/
STATE_FILE/parallelism settings in multiple cells with drifting values
(state_closed_v8.pkl, then state_closed_v4.pkl, then a third
state_closed_v8_merge.pkl snapshot). Here each constant is declared exactly
once.

Importing this module has no side effects beyond reading `.env` and
configuring logging -- no network/DB calls.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# repo layout is fixed: <repo_root>/graph_construction/pipeline/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_CONSTRUCTION_DIR = REPO_ROOT / "graph_construction"

load_dotenv(REPO_ROOT / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("graph_construction.pipeline")

# ── Target ITB / write toggle ─────────────────────────────────────────────────
TARGET_ITB_ID = os.getenv("TARGET_ITB_ID", "ROC_INPEX")

# ── Azure Search index this pipeline reads chunks from ───────────────────────
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "llm4t-documents-index-hok")

# ── Entity category ontology ──────────────────────────────────────────────────
ENTITY_CARDS_PATH = GRAPH_CONSTRUCTION_DIR / "entity_category_cards.json"

# ── Checkpointing ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR = GRAPH_CONSTRUCTION_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# The running state used throughout extraction + entity-merge.
STATE_FILE = CHECKPOINT_DIR / "extraction_state.pkl"
STATE_TMP_FILE = CHECKPOINT_DIR / "extraction_state.tmp.pkl"
# Explicit safety snapshot taken right after entity-description merging,
# before the Neo4j write stage (same purpose as the old state_closed_v8_merge.pkl).
STATE_POST_MERGE_FILE = CHECKPOINT_DIR / "extraction_state_post_merge.pkl"

CHECKPOINT_EVERY_N_CHUNKS = 5
CHECKPOINT_EVERY_N_FILES = 1

# ── Batch logs (per-batch extraction progress, written during run_parallel_extraction) ──
BATCH_LOG_DIR = GRAPH_CONSTRUCTION_DIR / "batch_logs"
BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Parallelism ────────────────────────────────────────────────────────────────
# Rule of thumb: FILE_BATCH_SIZE * avg_chunks_per_file < your APIM rate limit.
FILE_BATCH_SIZE = 8
MAX_WORKERS_PER_BATCH = 8
MAX_DESCRIPTION_MERGE_WORKERS = 10

# ── Entity description merging ────────────────────────────────────────────────
DESCRIPTION_MERGE_THRESHOLD = 70  # only merge entities seen 2..70 times
MAX_DESCRIPTION_WORDS = 100

# ── Relation embeddings ───────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE", "text-embedding-3-large-llm4t")
EMBED_DIMENSIONS = 3072
EMBED_BATCH = 256
