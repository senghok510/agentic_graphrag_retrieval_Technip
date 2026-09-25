# Knowledge Graph Construction — Architecture & Design

This document describes how the tender knowledge graph is built from raw ITB documents, covering the Neo4j schema, the LLM extraction prompt, entity category taxonomy, relation model, and the full pipeline flow.

Source notebook: `output/jupyter-notebook/extraction_pipeline.ipynb`

---

## 1. Neo4j Graph Schema

### Node Types

| Label | Key Property | Description |
|---|---|---|
| `ITB` | `ITB_ID` | Top-level container for a tender (e.g. `ROC_INPEX`) |
| `Document` | `docURL` | A single PDF or file within the ITB |
| `Chunk` | `ChunkID` | A text chunk extracted from a document (via Azure Document Intelligence) |
| `Entity` | `entityKey` | A named entity extracted from chunk text by the LLM |
| `Relation` | `relationId` | A semantic relationship between two entities |
| `HighLevelDomain` | `domainName` | Top engineering/business discipline tier — `TECHNICAL` or `NON-TECHNICAL` |
| `MidLevelDomain` | `domainName` | 10 mid-level disciplines (e.g. `ENGINEERING`, `LEGAL / CONTRACT`, `FINANCE / TAX`) — see Section 5 |
| `LowLevelDomain` | `domainName` | 44 fine-grained disciplines (e.g. `INSTRUMENTATION`, `PIPING INSTALLATION`, `HSE MANAGEMENT`) — see Section 5 |

### Relationships

| Pattern | Meaning |
|---|---|
| `(Chunk)-[:PART_OF]->(Document)` | A chunk belongs to a document |
| `(Document)-[:BELONGS_TO]->(ITB)` | A document is part of an ITB |
| `(Chunk)-[:MENTIONS]->(Entity)` | A chunk mentions an entity |
| `(Entity)-[:SUBJECT_OF]->(Relation)` | Entity is the subject of a relation |
| `(Relation)-[:OBJECT_OF]->(Entity)` | Entity is the object of a relation |
| `(Chunk)-[:ASSERTS {evidenceText, confidence, ...}]->(Relation)` | A chunk provides evidence for a relation, with full provenance |
| `(Document)-[:HAS_HIGH_DOMAIN]->(HighLevelDomain)` | Document-level classification into a top discipline tier |
| `(Document)-[:HAS_MID_LEVEL]->(MidLevelDomain)` | Document-level classification into a mid-level discipline |
| `(Document)-[:HAS_LOW_LEVEL]->(LowLevelDomain)` | Document-level classification into a fine-grained discipline (a document can have 0, 1, or several) |
| `(Relation)-[:CLASSIFIED_AS {reasoning}]->(LowLevelDomain \| MidLevelDomain)` | Relation-level classification into a discipline, with the LLM's one-line justification (Section 5) |

### Entity Node Properties

| Property | Type | Description |
|---|---|---|
| `entityKey` | string | SHA-1 hash of `entity\|\|{normalized_name}` — stable unique ID |
| `name` | string | Original entity name as extracted |
| `normalizedName` | string | Lowercased, punctuation-stripped form used for deduplication |
| `canonicalCategory` | string | One of the 32 predefined categories (see Section 3) |
| `llmCategory` | string | Free-form category proposed by the LLM (may differ from canonical) |
| `description` | string | Synthesized canonical description (LLM-merged across occurrences) |
| `aliases` | list[string] | Abbreviations or alternate names from the source text |

### Relation Node Properties

| Property | Type | Description |
|---|---|---|
| `relationId` | string | SHA-1 hash of `subject_key + relationship_label + object_key + description` |
| `relationshipLabel` | string | Free-form snake_case label (e.g. `requires_submission_of`, `subject_to`) |
| `relationshipDescription` | string | Full natural-language statement of the relationship |
| `relationshipStrength` | int | 1–10 score indicating importance in the source chunk |
| `weight` | float | `relationshipStrength / 10.0` |
| `supportCount` | int | Number of `ASSERTS` edges (computed after full write) |
| `avgConfidence` | float | Average LLM confidence across all asserting chunks |
| `aggregatedWeight` | float | `supportCount × (avgStrength / 10) × avgConfidence` |

### ASSERTS Edge Properties

| Property | Description |
|---|---|
| `evidenceText` | Exact text from the chunk supporting this relation |
| `confidence` | LLM confidence score (0.0–1.0) |
| `relationshipStrength` | Per-assertion strength (1–10) |
| `pageNumber` | Source page in the document |
| `docURL` | Source document URL |
| `fileName` | Source file path within the ITB |

---

## 2. LLM Extraction Prompt — Two-Stage Design

Each chunk is sent to the LLM in a single call. The prompt enforces **two sequential stages** before any entity or relation can be returned.

### Stage 1 — Chunk Cleaning

The LLM first classifies the chunk and removes structural noise before extracting anything.

**Cleaning outcomes:**

| Status | Meaning | Effect |
|---|---|---|
| `unchanged` | Chunk is already substantive | `cleaned_chunk = original_chunk`, `removed_spans = []` |
| `cleaned` | Mix of useful content and noise | `cleaned_chunk` keeps only substantive spans |
| `fully_removed` | No useful content remains | `entities = []`, `relations = []`, no Neo4j write |

**What gets removed:**
- Document cover/title-only lines
- Page number lines
- Repeated running headers and footers
- Table-of-contents navigation lines
- Pure separator/index lines

**What is always preserved:**
- All contractual, technical, operational, commercial, and legal wording
- Section headings when needed to interpret adjacent content
- Tables, bullets, and lists that carry substantive meaning

### Stage 2 — Semantic Extraction

Extraction runs **only on `cleaned_chunk`**, never on removed text.

Three things are extracted:

1. **Entities** — specific actors, objects, documents, requirements, values, roles, etc.
2. **Relationships** — semantic connections between entities as natural-language statements
3. **Evidence spans** — exact supporting text from `cleaned_chunk` for each relation

### Prompt Output Schema (strict JSON)

```json
{
  "cleaning_status": "unchanged | cleaned | fully_removed",
  "cleaning_reason": "short reason",
  "page_type": "substantive_clause | table_of_contents | cover_page | ...",
  "original_chunk": "...",
  "cleaned_chunk": "...",
  "removed_spans": ["span1", "span2"],
  "clause_summary": "one-line summary",
  "entities": [
    {
      "name": "Performance Test 3",
      "canonical_category": "test",
      "llm_category": "performance_test",
      "description": "...",
      "aliases": ["PT3"]
    }
  ],
  "relations": [
    {
      "subject": "Performance Test 3",
      "subject_canonical_category": "test",
      "subject_llm_category": "performance_test",
      "object": "Performance Test 1",
      "object_canonical_category": "test",
      "object_llm_category": "performance_test",
      "relationship_label": "can_only_commence_after",
      "relationship_description": "Performance Test 3 can only commence after Performance Test 1 has been successfully completed.",
      "relationship_strength": 9,
      "evidence": "Performance Test 3 can only commence after both PT1 and PT2 have been successfully completed.",
      "confidence": 0.94
    }
  ]
}
```

### Self-Check Rules (enforced in the prompt)

Before the LLM returns its JSON, the prompt mandates a built-in validation pass:

1. Every `canonical_category` must be in the 32-item list, or set to `"other"`.
2. Every relation `subject` and `object` must exactly match a name in the `entities` list — if not, the relation is dropped.
3. Every relation `evidence` must be a substring of `cleaned_chunk` — if not, the relation is dropped.
4. Every relation must have `relationship_label`, `relationship_description`, and `relationship_strength` — if any is missing, the relation is dropped.

---

## 3. Entity Category Taxonomy (32 Predefined Categories)

All entity categories are defined with definitions and examples in `output/jupyter-notebook/entity_category_cards.json`. The LLM is given the full card (definition + examples) for each category.

| Category | Meaning |
|---|---|
| `document` | A complete tender or contract file (ITB, GCC, specification, etc.) |
| `section` | A numbered or named subdivision within a document (schedule, exhibit, annex) |
| `clause` | A single atomic contractual provision (Clause 12.4, Article 8.1) |
| `test` | A defined verification activity with pass/fail criteria (FAT, SAT, PT) |
| `functional_group` | Grouping of systems/equipment by engineering function |
| `system` | A specific named engineering system (e.g. AGRU, CCS Compressor) |
| `equipment` | A specific tagged piece of equipment |
| `material` | A material, commodity, or bulk material specification |
| `requirement` | A stated technical or commercial requirement |
| `obligation` | A contractual duty or commitment ("shall", "must") |
| `condition` | A conditional trigger or prerequisite |
| `deliverable` | A work product to be submitted or handed over |
| `penalty` | A financial or contractual penalty (liquidated damages, etc.) |
| `approval` | An approval activity or formal sign-off event |
| `milestone` | A schedule milestone or key project date |
| `payment` | A payment event, installment, or payment term |
| `role` | A named role or function (e.g. Project Manager, Contractor) |
| `discipline` | An engineering or procurement discipline (Civil, Instrumentation, etc.) |
| `standard` | A referenced standard or code (ISO, AS/NZS, etc.) |
| `regulation` | A statutory or regulatory requirement |
| `permit` | A permit, license, or regulatory approval |
| `organization` | A named company, authority, or body |
| `location` | A geographic or facility location |
| `amount` | A monetary value or quantity |
| `bond` | A performance bond or financial security |
| `insurance` | An insurance requirement or policy |
| `duration` | A time period or schedule duration |
| `threshold` | A numeric threshold or limit |
| `term` | A defined contractual term or definition |
| `person` | A named individual |
| `project` | A named project or program |
| `risk` | A risk, hazard, or uncertainty |
| `other` | Anything that does not fit the above (LLM proposes its own `llm_category`) |

When no canonical category fits, the LLM sets `canonical_category = "other"` and preserves its own proposed type in `llm_category`. This allows post-hoc taxonomy extension without reprocessing.

---

## 4. Relation Model — Free-Form GraphRAG Style

Relations are **not constrained to a fixed ontology**. Instead the LLM produces:

- `relationship_label` — a short `snake_case` phrase (e.g. `requires_submission_of`, `subject_to`, `incorporates`, `defines`)
- `relationship_description` — a complete natural-language sentence preserving tender-specific wording (`shall`, `must`, `prior to`, `in accordance with`)
- `relationship_strength` — integer 1–10 scoring the importance relative to other relations in the same chunk (10 = the central obligation of the chunk)
- `evidence` — exact verbatim text from `cleaned_chunk`
- `confidence` — LLM self-reported confidence (0.0–1.0)

**Relation identity** is determined by the SHA-1 of `subject_key + relationship_label + object_key + description.lower()`. Two chunks asserting the same logical relation merge into a single `Relation` node with multiple `ASSERTS` edges.

---

## 5. Domain Classification Layer

Documents *and* individual relations are both classified against the same discipline taxonomy, so retrieval can be scoped by engineering domain at either granularity. Source: `chatbot/domain_data/domain_description.json` + `domain_keywords.json` (document-level taxonomy, also read at query time by `src/pipeline/domain_routing.py`) and `chatbot/relation_classification.ipynb` (relation-level classification).

### Taxonomy (3 tiers)

| Tier | Node label | Count | Examples |
|---|---|---|---|
| High | `HighLevelDomain` | 2 | `TECHNICAL`, `NON-TECHNICAL` |
| Mid | `MidLevelDomain` | 10 | `ENGINEERING`, `PROCUREMENT`, `PROCESS & HSED (SAFETY IN DESIGN)`, `PROJECT MANAGEMENT`, `CONSTRUCTION & COMM. START UP`, `LEGAL / CONTRACT`, `FINANCE / TAX`, `INSURANCE`, `INTELLECTUAL PROPERTY / NDA`, `GENERAL` |
| Low | `LowLevelDomain` | 44 | `INSTRUMENTATION`, `ELECTRICAL`, `PIPING INSTALLATION`, `ROTATING EQUIPMENT`, `HSE MANAGEMENT`, `DOCUMENT CONTROL`, `SCHEDULE MANAGEMENT`, … , `GENERAL` |

Each domain (every tier) carries a one-paragraph `domainDescription` (`domain_description.json`) and a short keyword list (`domain_keywords.json`). The keyword lists are **signals only** for the LLM prompt — both the document classifier and the relation classifier are instructed to classify by semantic meaning, never by keyword match alone.

### Document-level classification (existing)

Each `Document` is linked to its high/mid/low domain(s) via `HAS_HIGH_DOMAIN` / `HAS_MID_LEVEL` / `HAS_LOW_LEVEL` (Section 1). This is what `find_documents_by_domain()` and `get_document_details()` in `src/services/neo4j_service.py` traverse, and what `neo4j_vector_search()` reads back per chunk (`domain.domainName AS low_level_domain`) for context enrichment.

### Relation-level classification (`relation_classification.ipynb`)

Individual `Relation` nodes are **also** classified into the same taxonomy — a relation inherits candidate domains from its source document, but the classification itself is per-relation, not copied from the document:

1. **Load every relation with provenance.** For each file, pull every `Relation` asserted by its chunks, plus the subject/object entity names and descriptions:
   ```cypher
   MATCH (d:Document)<-[:PART_OF]-(c:Chunk)-[:ASSERTS]->(r:Relation)
   OPTIONAL MATCH (subj:Entity)-[:SUBJECT_OF]->(r)
   OPTIONAL MATCH (r)-[:OBJECT_OF]->(obj:Entity)
   RETURN d.fileName, r.relationId, r.relationshipDescription, r.relationshipLabel,
          c.text, subj.name, subj.description, obj.name, obj.description
   ```
2. **Two-tier candidate set per file.** If the document already has low-level domains assigned (`HAS_LOW_LEVEL`), the candidate list for every relation in that file is `{those low-level domains} + GENERAL`. If the document has **no** low-level domain, the candidate list falls back to `{the document's mid-level domains} + GENERAL`. This keeps the LLM's choice bounded to domains already plausible for that file, rather than the full 44/10-item taxonomy.
3. **One LLM call per relation**, using `CLASSIFY_LOWER_PROMPT_*` (or `CLASSIFY_MID_PROMPT_*` in the fallback case). The prompt gives the relationship label, subject/object entity names + descriptions, the source clause (truncated to 1200 chars), and the candidate domains' definitions (+ keyword hints), and asks the LLM to classify by the **relationship's meaning** — using the label and subject/object entities to disambiguate — returning strict JSON: `{"reasoning": "...", "lower_domain": "<domain or GENERAL>"}` (or `"mid_domain"` in the fallback case).
4. **Parallelized** across all `(file, relation)` pairs with a `ThreadPoolExecutor(max_workers=8)` — on the `ROC_INPEX` corpus this classifies ~6,800 relations. Result: one domain (low- or mid-level, whichever tier that file used) + a one-line LLM reasoning string per relation, keyed by `relationship_description`.
5. **Write-back to Neo4j.** Each classified relation is linked to its domain node with a `CLASSIFIED_AS` edge, batched 500 rows at a time via `UNWIND`. The match is deliberately **scoped to the source document** — it re-derives the `Relation` from `(Document)<-[:PART_OF]-(Chunk)-[:ASSERTS]->(Relation {relationshipDescription})` rather than matching on `relationshipDescription` alone, so two different documents that happen to assert relations with the same description text don't cross-link. The target domain node is reached through the document's *own* `HAS_LOW_LEVEL` / `HAS_MID_LEVEL` edge (not a bare domain-name lookup), which enforces that a relation can only be classified into a domain its own document was already assigned in Section 1 — consistent with the two-tier candidate set in step 2. The LLM's reasoning is kept on the edge:

   ```cypher
   // low-level tier
   UNWIND $rows AS row
   MATCH (d:Document {docURL: row.docURL})<-[:PART_OF]-(:Chunk)-[:ASSERTS]
         ->(r:Relation {relationshipDescription: row.description})
   MATCH (d)-[:HAS_LOW_LEVEL]->(l:LowLevelDomain {domainName: row.domain})
   MERGE (r)-[c:CLASSIFIED_AS]->(l)
   SET c.reasoning = row.reasoning
   RETURN count(*) AS linked

   // mid-level tier (documents with no low-level domain assigned)
   UNWIND $rows AS row
   MATCH (d:Document {docURL: row.docURL})<-[:PART_OF]-(:Chunk)-[:ASSERTS]
         ->(r:Relation {relationshipDescription: row.description})
   MATCH (d)-[:HAS_MID_LEVEL]->(m:MidLevelDomain {domainName: row.domain})
   MERGE (r)-[c:CLASSIFIED_AS]->(m)
   SET c.reasoning = row.reasoning
   RETURN count(*) AS linked
   ```

   The two Cypher statements run independently over `lower_rows` and `mid_rows` (the relations classified at each tier), each batched through `run_in_batches(session, cypher, rows, batch=500)`.

This gives a `LowLevelDomain` / `MidLevelDomain` node (e.g. `INSTRUMENTATION`) two kinds of incoming edges — `HAS_LOW_LEVEL` / `HAS_MID_LEVEL` from `Document` nodes (whole-file classification, Section 1) and `CLASSIFIED_AS` from individual `Relation` nodes (per-relation classification) — so retrieval can narrow to "just the `INSTRUMENTATION` relations" inside a mixed-domain document, not only filter at the whole-document level.

---

## 6. Entity Deduplication & Description Merging

### Deduplication

- Entity identity is determined by `SHA-1("entity||" + normalized_name)` where `normalized_name = lowercase(strip_non_alphanumeric(name))`.
- The first occurrence wins for all fields. Subsequent occurrences for the same key add new `MENTIONS` edges without overwriting the node.
- Aliases from later occurrences are **not merged** — only the first-seen aliases list is stored.

### Description Merging (LLM post-processing)

Entities that appear **2–70 times** across chunks get their per-chunk descriptions synthesized into a single canonical description by a second LLM call using the `ENTITY_DESCRIPTION_SUMMARY_PROMPT`.

The merge prompt instructs the LLM to:
- Create a fact-dense canonical definition from all collected descriptions
- Resolve conflicts (technical specifications take priority over general scope statements)
- Avoid meta-references ("the documents mention…")
- Stay within 100 words

The merged descriptions are written back to Neo4j via a batch Cypher `UNWIND … SET e.description = row.description` after extraction is complete.

---

## 7. Parallel Extraction Pipeline

### Parallelism Settings

```python
FILE_BATCH_SIZE = 8        # files processed concurrently per batch
MAX_WORKERS_PER_BATCH = 8  # threads (one per file in the batch)
```

### State Object (pickle checkpoint)

All accumulated data is kept in a single dictionary serialized to `checkpoints/state_closed_v4.pkl` after every batch:

| Key | Contents |
|---|---|
| `all_entity_nodes` | List of entity dicts |
| `all_relation_nodes` | List of relation dicts |
| `all_chunk_payloads` | List of chunk metadata dicts |
| `all_mentions` | List of `{chunk_id, entity_key}` pairs |
| `all_assertions` | List of full assertion records with evidence + provenance |
| `seen_entity_keys` | Set of already-written entity keys (dedup) |
| `seen_relation_ids` | Set of already-written relation IDs (dedup) |
| `completed_chunk_ids` | Set of chunk IDs already processed |
| `completed_files` | Set of file paths already processed |
| `failed_chunks` | List of failed chunk records for retry |
| `stats` | Running counters |

### Full Pipeline Flow

```
Azure AI Search
  (all chunks for ROC_INPEX ITB)
         │
         ▼
  Filter excluded files (14 files excluded — supplements, XLS, superseded docs)
         │
         ▼
  Parallel Extraction  ─── 8 files × threads
  for each chunk:
    1. LLM call (Stage 1: clean  →  Stage 2: extract entities + relations)
    2. Build entity_nodes, relation_nodes, mentions, assertions
    3. Merge into global state (thread-safe via state_lock)
         │
         ▼
  Checkpoint every batch  →  checkpoints/state_closed_v4.pkl
         │
         ▼
  Description Merge  (LLM synthesis for entities with 2–70 occurrences)
         │
         ▼
  Save merged state  →  checkpoints/state_closed_v5_merge.pkl
         │
         ▼
  Neo4j Write
    ├── Constraints + indexes
    ├── MERGE Entity nodes
    ├── MERGE Relation nodes + SUBJECT_OF / OBJECT_OF edges
    ├── MERGE Chunk→Entity  MENTIONS edges
    ├── MERGE Chunk→Relation  ASSERTS edges (with evidence + provenance)
    └── Aggregate supportCount / avgConfidence / aggregatedWeight on Relation nodes
