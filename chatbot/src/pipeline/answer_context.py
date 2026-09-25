"""Context assembly + citation utilities (Stage 6 helpers).

Verbatim from ppr.py: turns chunk-level evidence into file-level numbered
sources, and remaps sparse ``[Source N]`` citations onto a compact reference
list. Kept in its own module so both the retrieval branches and the answer
stage can import it without a cycle.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .schemas import ReferenceGeneration


def clean_whitespace(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _format_page_set(values) -> str:
    cleaned = []
    for value in values:
        text = str(value if value is not None else "N/A").strip() or "N/A"
        if text not in cleaned:
            cleaned.append(text)

    def _sort_key(text):
        return (0, int(text)) if text.isdigit() else (1, text)

    return ", ".join(sorted(cleaned, key=_sort_key)) if cleaned else "N/A"


def group_knowledge_base_by_file(chunks: list) -> list:
    """Convert chunk-level evidence into source-level (per unique file) evidence."""
    grouped, by_file = [], {}
    for chunk in chunks:
        file_name = chunk.get("file_name") or "Unknown"
        key = file_name.strip().lower()
        if key not in by_file:
            by_file[key] = {
                "file_name": file_name, "page_number": "N/A", "page_numbers": [],
                "content": "", "chunks": [], "source": "grouped_file", "score": 0.0,
            }
            grouped.append(by_file[key])
        source = by_file[key]
        page_number = str(chunk.get("page_number", "N/A"))
        source["page_numbers"].append(page_number)
        source["score"] = max(float(source.get("score") or 0.0), float(chunk.get("score") or 0.0))
        source["chunks"].append({
            "content": chunk.get("content", "") or chunk.get("text", ""),
            "page_number": page_number,
            "chunk_id": chunk.get("chunk_id", ""),
            "source": chunk.get("source", "unknown"),
            "score": chunk.get("score", 0.0),
        })
    for source in grouped:
        source["page_number"] = _format_page_set(source["page_numbers"])
        source["page_numbers"] = source["page_number"]
        source["content"] = "\n\n".join(
            f"[Page {chunk.get('page_number', 'N/A')} | Chunk {idx}]\n{chunk.get('content', '')}"
            for idx, chunk in enumerate(source["chunks"], start=1)
            if chunk.get("content")
        )
    return grouped


def build_numbered_context(knowledge_base: list) -> str:
    """Pre-number unique file sources and include their page/chunk evidence."""
    parts = []
    for i, source in enumerate(knowledge_base, start=1):
        fn = source.get("file_name", "Unknown")
        pages = source.get("page_numbers") or source.get("page_number", "N/A")
        if source.get("chunks"):
            snippet_parts = []
            for j, chunk in enumerate(source["chunks"], start=1):
                pn = chunk.get("page_number", "N/A")
                body = chunk.get("content", "")
                if body:
                    snippet_parts.append(f"[Page {pn} | Evidence {j}]\n{body}")
            body = "\n\n".join(snippet_parts)
        else:
            body = source.get("content", str(source))
        parts.append(f"[Source {i}] (File: {fn}, Pages: {pages})\n{body}")
    return "\n\n".join(parts)


def _reference_for(knowledge_base: list, idx: int) -> ReferenceGeneration:
    entry = knowledge_base[idx - 1]
    return ReferenceGeneration(
        file_name=entry.get("file_name", "Unknown"),
        page_number=str(entry.get("page_numbers") or entry.get("page_number", "N/A")),
    )


def remap_citations(answer: str, knowledge_base: list,
                    fallback_top_n: int = 5) -> Tuple[str, List[ReferenceGeneration]]:
    """Remap sparse ``[Source N]`` citations and build a compact reference list.

    Accepts both the mandated ``[Source N]`` / ``[Sources N, M]`` form and a bare
    ``[N]`` / ``[N, M]`` bracket-number form — models frequently abbreviate to the
    latter (e.g. in bullet lists) despite the prompt asking for the former. The
    bracket content must look like a citation list (digits + separators only) so
    unrelated bracketed numbers (years, percentages, units) are not swept in.

    If the answer contains no recognisable citation at all, falls back to citing
    the top ``fallback_top_n`` knowledge-base sources (already relevance-ranked by
    the upstream cross-encoder) so the UI is never left with an empty source list.
    """
    citation_pattern = re.compile(
        r"\[\s*(?:Sources?\s+)?(\d+(?:\s*(?:,|and|&)\s*(?:Sources?\s+)?\d+)*)\s*\]", re.IGNORECASE)
    cited_numbers = []
    for match in citation_pattern.finditer(answer):
        for raw_num in re.findall(r"\d+", match.group(1)):
            num = int(raw_num)
            if 1 <= num <= len(knowledge_base) and num not in cited_numbers:
                cited_numbers.append(num)

    if not cited_numbers:
        # Model didn't cite inline at all (or used a format we don't recognise) —
        # still surface what it was actually grounded in rather than showing nothing.
        top_n = min(fallback_top_n, len(knowledge_base))
        references = [_reference_for(knowledge_base, i) for i in range(1, top_n + 1)]
        return answer, references

    mapping = {old: new for new, old in enumerate(cited_numbers, start=1)}

    def _replace(match):
        remapped = []
        for raw_num in re.findall(r"\d+", match.group(1)):
            new = mapping.get(int(raw_num))
            if new is not None and new not in remapped:
                remapped.append(new)
        if not remapped:
            # Explicit "[Source N]" that turned out unmappable (e.g. hallucinated,
            # out-of-range) is a broken citation — drop it. A bare "[N]" that never
            # mapped is probably not a citation at all (a year, a quantity, a list
            # marker) — leave the original text untouched rather than deleting it.
            return "" if "source" in match.group(0).lower() else match.group(0)
        return "[Source " + ", Source ".join(str(num) for num in remapped) + "]"

    remapped_answer = citation_pattern.sub(_replace, answer)
    references = [_reference_for(knowledge_base, old) for old in cited_numbers]
    return remapped_answer, references
