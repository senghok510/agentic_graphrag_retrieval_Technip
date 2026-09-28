"""Pulls chunk documents from Azure AI Search and resolves each file to its
Neo4j Document.docURL. Both functions take their client/driver explicitly --
call them from run_pipeline.py, not at import time.
"""

from __future__ import annotations

import logging
from typing import Any

from .text_utils import escape_odata_string, extract_page_and_chunk

logger = logging.getLogger("graph_construction.pipeline.search_ingest")

_SELECT_FIELDS = [
    "chunk_id",
    "tender_id",
    "file_name",
    "page_chunk",
    "blob_url",
    "chunk_index",
    "page_number",
    "chunk_type",
    "is_active",
    "high_level_domain",
    "mid_level_domain",
    "low_level_domain",
]


def fetch_all_files_docs(
    search_client,
    tender_id: str | None = None,
    batch_size: int = 1000,
) -> dict[str, list[dict[str, Any]]]:
    """All chunks in the index, grouped by file_name, sorted by (page, chunk)
    and stamped with a per-file sequence_number."""
    tender_filter = f"tender_id eq '{escape_odata_string(tender_id)}'" if tender_id else None
    facet_results = search_client.search(
        search_text="*",
        filter=tender_filter,
        facets=["file_name,count:200"],
        top=0,
    )
    all_file_names = [f["value"] for f in facet_results.get_facets().get("file_name", [])]
    logger.info("Total files in index: %d", len(all_file_names))

    all_files_docs: dict[str, list[dict[str, Any]]] = {}

    for file_name in all_file_names:
        docs: list[dict[str, Any]] = []
        skip = 0
        while True:
            file_filter = f"file_name eq '{escape_odata_string(file_name)}'"
            if tender_filter:
                file_filter = f"{tender_filter} and {file_filter}"
            results = search_client.search(
                search_text="*",
                filter=file_filter,
                select=_SELECT_FIELDS,
                top=batch_size,
                skip=skip,
            )
            batch = list(results)
            if not batch:
                break
            docs.extend(batch)
            if len(batch) < batch_size:
                break
            skip += batch_size
            if skip >= 100000:
                break

        sorted_docs = sorted(docs, key=lambda d: extract_page_and_chunk(d["chunk_id"]))
        for idx, doc in enumerate(sorted_docs, start=1):
            doc["sequence_number"] = idx
        all_files_docs[file_name] = sorted_docs

        logger.info(
            "  [%d/%d] %s: %d chunks",
            len(all_files_docs),
            len(all_file_names),
            file_name,
            len(docs),
        )

    total_chunks = sum(len(v) for v in all_files_docs.values())
    logger.info("Done. %d files, %d total chunks", len(all_files_docs), total_chunks)
    return all_files_docs


def _resolve_doc_url(azure_file_name: str, doc_url_list: list[str]) -> str | None:
    """azure_file_name: "ROC_INPEX/ITT_.../filename.pdf"
    docURL:            "file://c:\\Users\\...\\ROC_INPEX\\ITT_...\\filename.pdf"
    Match by normalizing slashes and checking suffix."""
    normalized = azure_file_name.replace("/", "\\")
    for url in doc_url_list:
        if url.replace("/", "\\").endswith(normalized):
            return url
    return None


def build_file_name_url_map(neo4j_driver, file_names: list[str]) -> dict[str, str | None]:
    """Resolves every Azure Search file_name to its Neo4j Document.docURL (or
    None if unresolved). Fetches the docURL list once, not once per file."""
    with neo4j_driver.session() as session:
        doc_url_list = [
            record["doc_url"]
            for record in session.run("MATCH (d:Document) RETURN d.docURL AS doc_url")
            if record["doc_url"]
        ]
    logger.info("Loaded %d docURLs from Neo4j", len(doc_url_list))

    return {file_name: _resolve_doc_url(file_name, doc_url_list) for file_name in file_names}
