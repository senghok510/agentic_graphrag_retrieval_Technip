"""Every prompt the pipeline sends to the LLM, plus the entity-category
ontology loader they're built from.

Loading entity_category_cards.json is lazy (@lru_cache) rather than eager at
import time -- keeps this module side-effect-free to import, consistent with
the rest of the package.
"""

from __future__ import annotations

import json
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def load_entity_cards() -> dict[str, dict]:
    return json.loads(config.ENTITY_CARDS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def canonical_entity_categories() -> list[str]:
    return list(load_entity_cards().keys())


def render_entity_category_block(cards: dict[str, dict]) -> str:
    blocks = []
    for name, c in cards.items():
        ex = ", ".join(c.get("examples", []))
        blocks.append(f"- {name}\n    definition : {c['definition']}\n    examples   : {ex}")
    return "\n".join(blocks)


def build_semantic_extraction_prompt(dynamic_entity_category_block: str) -> str:
    return f"""
You are extracting tender-contract semantics into a knowledge graph.

Your task has 2 stages:

STAGE 1 — CHUNK CLEANING

The input chunk may contain both useful semantic content and structural noise.

You must clean the chunk before semantic extraction.

Possible cleaning outcomes:
- unchanged
  - the chunk is already substantive and should be kept as-is
- cleaned
  - the chunk contains both useful content and removable structural noise
- fully_removed
  - the chunk contains no useful semantic content after cleaning

Preserve:
- original meaning
- contractual and technical wording
- substantive legal, technical, operational, commercial, testing, compliance, evidentiary, procedural, and scope content
- section headings or labels when they are needed to interpret substantive content
- substantive content whether it appears in prose, bullets, lists, or tables

You must remove only clearly structural or non-substantive noise such as:
- document cover/title-only lines
- page number lines
- repeated running headers/footers
- separator lines
- repeated navigation labels that do not introduce substantive content

Important:
- A chunk may contain both noise and useful information.
- Do NOT exclude the whole chunk just because some lines are structural.
- Remove only the irrelevant spans and keep the substantive spans.
- Keep headings, labels, row labels, and section labels if they are needed to preserve meaning.
- Remove a heading or label only if it is acting purely as navigation or index text.


STAGE 2 — SEMANTIC EXTRACTION

Extract semantics only from cleaned_chunk, never from removed text.

Extract:
1. Entities
   - The specific actors, objects, documents, sections, requirements, values, roles, organisations, systems, locations, dates, or clauses involved.

2. Relationships
   - The semantic connections between extracted entities.
   - Relationship types are free-form.
   - Express each relationship as a complete, faithful natural-language statement.

3. Evidence spans
   - The exact text from cleaned_chunk supporting each extracted entity or relationship.


ENTITY EXTRACTION RULES

You MUST:
- reuse the canonical entity categories when they fit
- ALWAYS fill llm_category with your proposed category, even if it matches the canonical category
- if no canonical category fits, set canonical_category to "other" and preserve your proposed category type in llm_category
- only extract entities explicitly supported by cleaned_chunk
- include aliases when the source provides abbreviations, alternate names, or bracketed names
- entity description must explain the entity's role, purpose, meaning, or function in the source chunk, according to its category
- do not merely restate the entity name in the description
- description must never be empty
- if the source gives enough context, ground the description in that context
- if the source only provides a list/table entry with no explanation, write: "Listed as a [canonical_category] in the source chunk under [nearby heading/table/row context]."
Canonical entity categories with definitions and examples:
{dynamic_entity_category_block}


RELATIONSHIP EXTRACTION RULES

You MUST:
- extract relationships only between entities that appear in the entities list
- preserve directional meaning where the source implies direction
- write relationship_label as a short free-form snake_case phrase describing the relationship
- write relationship_description as a complete natural-language statement
- preserve tender-specific wording such as shall, must, requires, prior to, listed in, submitted by, approved by, incorporated into, applies to, included in, responsible for, subject to, and in accordance with when present
- prefer precise contractual, technical, procedural, commercial, schedule, approval, responsibility, inclusion, reference, or definition statements
- avoid generic relationships such as "related to" unless the source provides no clearer relationship
- include exact supporting evidence from cleaned_chunk
- assign relationship_strength as an integer from 1 to 10
- write relationship_keywords as 3-6 snake_case high-level theme tags summarizing what KIND of relationship this is (e.g. schedule_dependency, payment_terms, compliance_obligation, liability_allocation, scope_inclusion, test_sequencing, approval_workflow, document_reference) — never entity names, clause numbers, or specific values

Relationship strength / weight guidance:
Use the full 1-10 range. Score relationship strength relative to the other relationships in the same chunk.

- 10: Reserved for the central relationship that defines the main meaning of the chunk, such as the primary obligation, direct definition, direct listing entry, or explicit prerequisite.
- 8-9: Strong explicit relationship that is directly stated and important, but not the only central point of the chunk.
- 6-7: Clear supporting relationship, such as inclusion, reference, applicability, responsibility, schedule linkage, or document-to-document linkage.
- 4-5: Contextual relationship that is useful but secondary or indirectly stated.
- 1-3: Incidental co-occurrence or weak association. Usually do not extract these.

Normalization rules:
- Use snake_case for canonical_category, llm_category, and relationship_label.
- If there are no useful entities or relationships, return empty lists.
- llm_category is ALWAYS filled when entities are returned.
- subject and object in each relation must exactly match names from the entities list.


─────────────────────────────────────────────────────────────────────────
SELF-CHECK BEFORE RETURNING
─────────────────────────────────────────────────────────────────────────
For every entity and relation in your output, verify:
1. Every entity canonical_category is in the canonical entity list above, or is exactly "other".
   If not → set canonical_category to "other" and keep your proposed type in llm_category.

2. Every relation subject and object exactly match an entity name in the entities list.
   If not → drop the relation.

3. Every relation evidence is a substring of cleaned_chunk.
   If not → drop the relation.

4. Every relation has relationship_label, relationship_description, relationship_strength, and relationship_keywords.
   If not → drop the relation.


Return STRICT JSON only with this schema:

{{
  "cleaning_status": "unchanged" | "cleaned" | "fully_removed",
  "cleaning_reason": "short reason",
  "page_type": "substantive_clause | substantive_appendix | table_of_contents | appendix_index | cover_page | revision_history | structural_navigation | other",
  "original_chunk": "the original input chunk exactly as received",
  "cleaned_chunk": "the cleaned chunk used for extraction; empty string if fully_removed",
  "removed_spans": [
    "exact removed span 1",
    "exact removed span 2"
  ],
  "clause_summary": "short summary",
  "entities": [
    {{
      "name": "Performance Test 3",
      "canonical_category": "test",
      "llm_category": "performance_test",
      "description": "Performance Test 3 is a defined verification activity referenced in the source chunk.",
      "aliases": ["PT3"]
    }}
  ],
  "relations": [
    {{
      "subject": "Performance Test 3",
      "subject_canonical_category": "test",
      "subject_llm_category": "performance_test",
      "object": "Performance Test 1",
      "object_canonical_category": "test",
      "object_llm_category": "performance_test",
      "relationship_label": "can_only_commence_after",
      "relationship_description": "Performance Test 3 can only commence after Performance Test 1 has been successfully completed.",
      "relationship_strength": 9,
      "relationship_keywords": ["test_sequencing", "prerequisite_completion"],
      "evidence": "Performance Test 3 can only commence after both Performance Test 1 and Performance Test 2 have been successfully completed.",
      "confidence": 0.94
    }}
  ]
}}

Rules for cleaning outcomes:
- If cleaning_status = "unchanged":
  - cleaned_chunk should be the same as original_chunk
  - removed_spans should be []
- If cleaning_status = "cleaned":
  - cleaned_chunk should keep only substantive content
  - removed_spans should list what was removed
- If cleaning_status = "fully_removed":
  - cleaned_chunk must be ""
  - entities must be []
  - relations must be []
  - clause_summary should briefly say why nothing substantive remained

Example fully removed output:
{{
  "cleaning_status": "fully_removed",
  "cleaning_reason": "The chunk contains only table of contents and navigation text.",
  "page_type": "table_of_contents",
  "original_chunk": "3 PERFORMANCE TESTS .... 3\\n4 REMEDIES .... 6",
  "cleaned_chunk": "",
  "removed_spans": [
    "3 PERFORMANCE TESTS .... 3",
    "4 REMEDIES .... 6"
  ],
  "clause_summary": "No substantive contractual content remained after cleaning.",
  "entities": [],
  "relations": []
}}

Example cleaned output:
{{
  "cleaning_status": "cleaned",
  "cleaning_reason": "Removed structural navigation lines and retained substantive testing requirements.",
  "page_type": "substantive_clause",
  "original_chunk": "3 PERFORMANCE TESTS .... 3\\nPerformance Test 3 can only commence after both Performance Test 1 and Performance Test 2 have been successfully completed.",
  "cleaned_chunk": "Performance Test 3 can only commence after both Performance Test 1 and Performance Test 2 have been successfully completed.",
  "removed_spans": [
    "3 PERFORMANCE TESTS .... 3"
  ],
  "clause_summary": "Performance Test 3 has a prerequisite relationship to Performance Test 1 and Performance Test 2.",
  "entities": [
    {{
      "name": "Performance Test 3",
      "canonical_category": "test",
      "llm_category": "performance_test",
      "description": "Performance Test 3 is a defined test activity that depends on prior successful completion of Performance Test 1 and Performance Test 2.",
      "aliases": []
    }},
    {{
      "name": "Performance Test 1",
      "canonical_category": "test",
      "llm_category": "performance_test",
      "description": "Performance Test 1 is listed in the source as a prerequisite test that must be successfully completed before Performance Test 3 can commence.",
      "aliases": []
    }},
    {{
      "name": "Performance Test 2",
      "canonical_category": "test",
      "llm_category": "performance_test",
      "description": "Performance Test 2 is listed in the source as a prerequisite test that must be successfully completed before Performance Test 3 can commence.",
      "aliases": []
    }}
  ],
  "relations": [
    {{
      "subject": "Performance Test 3",
      "subject_canonical_category": "test",
      "subject_llm_category": "performance_test",
      "object": "Performance Test 1",
      "object_canonical_category": "test",
      "object_llm_category": "performance_test",
      "relationship_label": "can_only_commence_after",
      "relationship_description": "Performance Test 3 can only commence after Performance Test 1 has been successfully completed.",
      "relationship_strength": 9,
      "relationship_keywords": ["test_sequencing", "prerequisite_completion"],
      "evidence": "Performance Test 3 can only commence after both Performance Test 1 and Performance Test 2 have been successfully completed.",
      "confidence": 0.94
    }},
    {{
      "subject": "Performance Test 3",
      "subject_canonical_category": "test",
      "subject_llm_category": "performance_test",
      "object": "Performance Test 2",
      "object_canonical_category": "test",
      "object_llm_category": "performance_test",
      "relationship_label": "can_only_commence_after",
      "relationship_description": "Performance Test 3 can only commence after Performance Test 2 has been successfully completed.",
      "relationship_strength": 9,
      "relationship_keywords": ["test_sequencing", "prerequisite_completion"],
      "evidence": "Performance Test 3 can only commence after both Performance Test 1 and Performance Test 2 have been successfully completed.",
      "confidence": 0.94
    }}
  ]
}}

Now clean this chunk and extract semantics from cleaned_chunk only.
"""


# ── Entity description merging ────────────────────────────────────────────────

ENTITY_DESCRIPTION_SUMMARY_PROMPT = """
You are a senior domain expert in EPC procurement and knowledge graph curation.
Your task is to synthesize multiple descriptions of the same entity into a single, canonical definition.

### OBJECTIVES:
1. **Canonical Synthesis:** Create a definitive, high-fidelity description that captures the entity's functional, technical, and contractual identity.
2. **Fact-Density:** Remove redundant phrasing. Focus on unique attributes, specific roles, and technical parameters present across the source descriptions.
3. **Conflict Resolution:** If descriptions conflict, prioritize technical specifications, performance requirements, or precise contractual obligations over general scope statements.
4. **Grounding:** Do not invent facts, roles, requirements, relationships, or technical parameters that are not present in the extracted descriptions.
5. **Formatting:** Write in a formal, objective, third-person style. Do not use meta-references such as "the documents state", "the descriptions mention", or "the source says".
6. **Constraint:** Max {max_length} words.

#######
Entity: {entity_name}

Extracted Descriptions:
{description_list}
#######

Return STRICT JSON only:
{{
  "description": "synthesized canonical description"
}}
""".strip()


# ── High-level relation keywords (backfill-only, see relation_keywords.py) ────

HIGH_LEVEL_PROMPT_SYS = """You are an expert in tendering and EPC/ITB contract documents.
You extract high-level thematic keywords that describe the OVERARCHING nature of a
relationship between two entities — concepts and themes, NOT the specific entities,
numbers, or document names involved."""

HIGH_LEVEL_PROMPT_USER = """Extract relationship_keywords from the relationship below.

relationship_keywords: 3-6 high-level theme tags that summarize what KIND of
relationship this is and WHY it matters in a tender/EPC context. Focus on concepts
and themes (e.g. schedule_dependency, payment_terms, compliance_obligation,
liability_allocation, scope_inclusion, test_sequencing, approval_workflow,
document_reference), NOT the specific entities, tags, clause numbers, or values.

Rules:
- Return snake_case, lowercase tags.
- Do NOT repeat entity names from the description as keywords.
- Prefer reusable themes that would group many similar relationships together.

Relationship description:
{relationship_description}

Source chunk (context only):
{chunk}

Return STRICT JSON only:
{{"relationship_keywords": ["theme_one", "theme_two"]}}
"""
