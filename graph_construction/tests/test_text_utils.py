from __future__ import annotations

from graph_construction.pipeline.text_utils import (
    entity_key_from_name,
    extract_page_and_chunk,
    normalize_entity_name,
    normalize_keywords,
    resolve_section_hint,
)


def test_chunk_identifiers_sort_by_page_then_chunk():
    identifiers = [
        "doc_page_10_chunk_1",
        "doc_page_2_chunk_3",
        "doc_page_2_chunk_1",
    ]
    assert sorted(identifiers, key=extract_page_and_chunk) == [
        "doc_page_2_chunk_1",
        "doc_page_2_chunk_3",
        "doc_page_10_chunk_1",
    ]


def test_entity_keys_are_stable_across_punctuation_and_case():
    assert normalize_entity_name("  Baker-Hughes! ") == "baker hughes"
    assert entity_key_from_name("Baker-Hughes!") == entity_key_from_name("baker hughes")


def test_keyword_normalization_deduplicates_and_falls_back_to_label():
    assert normalize_keywords(["Safety Case", "safety-case", "Permit"], cap=2) == [
        "safety_case",
        "permit",
    ]
    assert normalize_keywords([], label="requires_submission") == ["requires_submission"]


def test_section_hint_accepts_headings_only():
    assert resolve_section_hint("3.2 PERFORMANCE TESTS\nRequirements follow") == (
        "3.2 PERFORMANCE TESTS"
    )
    assert resolve_section_hint("This is a normal sentence.\nMore text") is None
