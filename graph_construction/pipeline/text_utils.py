"""Small, pure string/hash helpers shared across the pipeline. No I/O."""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Sequence, Tuple, Union


def extract_page_and_chunk(doc_id: str) -> Tuple[Union[int, str], int]:
    """Sortable (page, chunk) tuple.

    - PDF / Word: (page_number, chunk_number)
    - Excel:      (sheet_name, chunk_number)

    Expected ID formats:
      *_page_12_text_chunk_3  (PDFs with text chunks)
      *_page_12_chunk_3       (alternative format)
      *_page_Sheet1_chunk_7   (Excel sheets)
    """
    chunk_match = re.search(r"_chunk_(\d+)$", doc_id)
    chunk = int(chunk_match.group(1)) if chunk_match else 0

    page_match = re.search(r"_page_([^_]+?)(?:_text)?_chunk_", doc_id)
    page = page_match.group(1) if page_match else ""

    if page.isdigit():
        return (int(page), chunk)
    return (page.lower(), chunk)


def escape_odata_string(value: str) -> str:
    """Escapes single quotes for OData filters. O'Brien -> O''Brien."""
    return value.replace("'", "''")


def resolve_section_hint(chunk_text: str) -> Optional[str]:
    """First line of a chunk, if it looks like a numbered or all-caps heading."""
    section_hint = None
    first_line = chunk_text.splitlines()[0].strip() if chunk_text else ""
    if first_line and len(first_line) <= 140:
        if re.match(r"^(\d+(\.\d+)*)\s+.+", first_line):
            section_hint = first_line
        elif first_line.isupper() and len(first_line) <= 100:
            section_hint = first_line
    return section_hint


def normalize_entity_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower()).strip()


def entity_key_from_name(name: str) -> str:
    """SHA-1("entity||" + normalized_name) -- the stable Entity node key."""
    return hashlib.sha1(f"entity||{normalize_entity_name(name)}".encode("utf-8")).hexdigest()


def normalize_keywords(kws: Optional[Sequence[str]], label: Optional[str] = None, cap: int = 6) -> List[str]:
    """Lowercase, snake_case, de-duplicated keyword list, capped at `cap`.
    Falls back to `[label]` if nothing survives normalization (e.g. so a
    relation always has at least one theme tag)."""
    out: List[str] = []
    for kw in kws or []:
        if not isinstance(kw, str):
            continue
        k = re.sub(r"[^a-z0-9 ]+", " ", kw.lower()).strip().replace(" ", "_")
        if k:
            out.append(k)
    out = list(dict.fromkeys(out))[:cap]
    if not out and label:
        out = [label]
    return out
