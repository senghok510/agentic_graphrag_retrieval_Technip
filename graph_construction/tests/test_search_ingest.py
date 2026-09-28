from __future__ import annotations

from graph_construction.pipeline.search_ingest import fetch_all_files_docs


class _FacetResult:
    def get_facets(self):
        return {"file_name": [{"value": "Tender/O'Brien.pdf"}]}


class _SearchClient:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("facets"):
            return _FacetResult()
        if kwargs["skip"] == 0:
            return [
                {"chunk_id": "doc_page_2_chunk_1"},
                {"chunk_id": "doc_page_1_chunk_2"},
            ]
        return []


def test_ingest_scopes_every_search_to_tender_and_sequences_chunks():
    client = _SearchClient()

    result = fetch_all_files_docs(client, tender_id="ROC_INPEX", batch_size=1000)

    assert client.calls[0]["filter"] == "tender_id eq 'ROC_INPEX'"
    assert client.calls[1]["filter"] == (
        "tender_id eq 'ROC_INPEX' and file_name eq 'Tender/O''Brien.pdf'"
    )
    chunks = result["Tender/O'Brien.pdf"]
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "doc_page_1_chunk_2",
        "doc_page_2_chunk_1",
    ]
    assert [chunk["sequence_number"] for chunk in chunks] == [1, 2]
