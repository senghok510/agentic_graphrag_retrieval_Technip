# LightRAG Eval — Gold Expansion: Why and How

## TL;DR

The synthetic eval set (`eval_singlehop_lightrag_v2.jsonl`) has a structural bias in its
`global`-mode rows: the gold answer key was built from an **exact string match** on
`relationship_label`, while LightRAG retrieves by **embedding similarity** which ignores that
label. As a result, LightRAG gets penalized for retrieving relations that correctly answer the
question but happen to carry a different label. `expand_lightrag_gold.py` fixes the existing
data by adding those missing relations to the gold key; `generate_lightrag_eval.py` was patched
to prevent the problem in future generations.

**Expansion changes the answer key, not the retrieval.** LightRAG's behavior is untouched —
only what counts as "correct" when scoring it.

---

## 1. The problem

### How global-mode gold was built

`generate_lightrag_eval.py` creates a "global" (thematic/aggregation) question by:

1. Grouping every relation in the graph by its raw `relationship_label` string:

   ```python
   rels_by_label = defaultdict(list)
   for rid, r in relation_by_id.items():
       lbl = (r.get("relationship_label") or "").strip()   # exact string match
       rels_by_label[lbl].append(rid)
   ```

2. Picking one label bucket (e.g. `required_under`) as the "theme".
3. Asking an LLM to write a question + reference answer from those relations.
4. Setting `gold_relation_ids` = the relations from **that bucket only**.

### How LightRAG actually retrieves

```
question → hl-keywords (LLM) → embed → vector search over ALL relation embeddings → top-k
```

The retriever never sees `relationship_label`. It finds whatever is semantically close to the
query — regardless of which label bucket a relation landed in at graph-build time.

### Why these two collide

The graph-build LLM labels relations freely, so **the same kind of fact — even the exact same
fact — gets different labels on different extraction passes**. The theme bucket is therefore an
arbitrary subset of the true answer set.

---

## 2. A real example from this dataset

Question (row 6, `theme_label: "required_under"`):

> *"What are all the regulatory approvals and plans required under specific legislative acts
> for the project, including their associated dates and responsible authorities?"*

Gold contains 7 relations, all labeled `required_under`. But the graph also contains:

| relation_id | fact | label | in gold? |
|---|---|---|---|
| `57309dd8` | NT Environmental Approval (referral) → Environmental Protection Act (NT) | `required_under` | ✅ yes |
| `6c544bc0` | NT Environmental Approval (referral) → Environmental Protection Act (NT) | `governed_by` | ❌ no |

**Rows 1 and 2 are the same fact about the same entities from the same Exhibit K table** —
extracted twice with different labels, producing two relation IDs. Only the `required_under`
copy is gold.

When this question was run through the retrieval pipeline, LightRAG's global channel surfaced
(all with high `global_score`):

- `6c544bc0` — NT Environmental Approval **governed_by** Environmental Protection Act (NT)
- `30e27a69` — Development Consent – LNG Plant **regulated_by** Planning Act (NT)
- `c35a4f06` — Heritage Approval (for Surveys) **governed_by** Heritage Act 2011 (NT)
- `d3fcd58a` — Environment Protection Approval – Off Plot **governed_by** WMPC Act (NT)
- `055d300e` — Operational EMP (revision) **required_under** WMPC Act — ranked **#1**
  (`final_score: 1.0`), yet excluded from gold because the generator's own answer-writing LLM
  dropped it from the "required" set.

Every one of these is a regulatory approval required under a named legislative act — a correct
answer to the question. Under the original gold, **every one of them is scored as a retrieval
error**. Precision looks bad; the retriever was actually right.

So the observation "LightRAG can't retrieve the gold with this vague question" was backwards:
LightRAG retrieves the right *facts* fine — the *answer key* doesn't recognize them.

---

## 3. The fix for existing data: `expand_lightrag_gold.py`

For each `global` row (local rows pass through unchanged — their gold is a real 1-hop
neighborhood, not a label bucket):

```
row's gold_high_level_keywords
        │  embed (same style of query LightRAG's global channel uses)
        ▼
cosine similarity vs ALL ~6.7k evidenced relations in the graph
        │  keep top-20 that are NOT already gold and score ≥ 0.35
        ▼
LLM judge: "would this fact be one of the ENUMERATED ITEMS
            if the reference answer were exhaustive?"
        │  accept / reject per candidate (strict: reject same-topic
        │  background facts; when unsure, reject)
        ▼
accepted relations merged into gold_relation_ids,
gold_chunk_ids, gold_entity_keys
+ a `gold_expansion` audit field recording what was added
```

### Guardrails (why this isn't "just adding more relations")

1. **Candidate pool = what a retriever could plausibly find.** Candidates come from embedding
   similarity to the question's own hl-keywords — the same mechanism LightRAG uses. Relations
   semantically far from the question are never considered.
2. **LLM judge filters against the reference answer.** Only facts that belong as enumerated
   items get in. Early smoke-testing showed the first prompt was too permissive (it accepted
   `Contractor must_comply_with HSE regulatory obligations` for a question about *items that
   must be provided*), so the prompt now demands same-kind-of-fact matching and defaults to
   reject.
3. **Nothing is removed and the input file is never modified.** Output goes to a new file;
   every touched row carries a `gold_expansion` field for auditing:

   ```json
   "gold_expansion": {
     "candidates_considered": 20,
     "added_relation_ids": ["489608a4769d730d92791a9c7f726c5f88da501e"]
   }
   ```

### Smoke-test results (first 3 global rows, permissive v1 prompt)

| row | question theme | added | verdict |
|---|---|---|---|
| 1 | items that must be provided to Company/workers | +2 | ❌ too generous — compliance obligations, not provided items (prompt tightened since) |
| 3 | training requirements for HSES personnel | +2 | ✅ genuine competency/certification requirements |
| 5 | mandatory inclusions in project plans | +1 | ✅ `CEMP must_include Liquid Discharge Management Plan` — textbook label-gap case |

### Usage

```bash
# smoke test (first 3 global rows; relation embeddings cached after first run)
python chatbot/data/expand_lightrag_gold.py \
    --in  chatbot/data/eval_singlehop_lightrag_v2.jsonl \
    --out chatbot/data/eval_singlehop_lightrag_v2_expanded.jsonl \
    --limit 3

# full run (all 50 global rows)
python chatbot/data/expand_lightrag_gold.py \
    --in  chatbot/data/eval_singlehop_lightrag_v2.jsonl \
    --out chatbot/data/eval_singlehop_lightrag_v2_expanded.jsonl
```

Knobs: `--top-n` (max candidates judged per row, default 20), `--sim-floor` (min cosine to be a
candidate, default 0.35), `--limit` (rows to expand; the rest pass through).

---

## 4. Effect on metrics

Scoring retrieval as set-overlap between retrieved relation IDs and `gold_relation_ids`:

| metric | before expansion | after expansion |
|---|---|---|
| **precision** | understated — correct retrievals like `6c544bc0` count as errors | ↑ false penalties disappear |
| **recall** | overstated — gold is missing valid answers, so "full recall" was easy | may ↓ — but a miss on an added relation is now a *real* miss (the question asks for **all** approvals) |

The before/after delta on global rows is itself useful: it quantifies how much the
label-bucketing artifact was distorting the numbers.

The scoring itself lives in the last 4 cells of `chatbot/evaluate_lightrag.ipynb`
(`run_lightrag_pipeline`, `evaluate_lightrag_dataset`) — point `EVAL_SET_PATH` at the expanded
file. Running it against both the original and expanded files gives the comparison directly.

---

## 5. The root-cause fix for future data: label clustering in the generator

Expanding gold treats the symptom in already-generated data. The cause — themes formed by
exact label strings — is fixed in `generate_lightrag_eval.py`: raw `relationship_label`s are
now embedded and greedily clustered by cosine similarity
(`LABEL_CLUSTER_SIM_THRESHOLD = 0.55`) **before** theme bucketing, so near-synonym labels like
`required_under` / `governed_by` / `regulated_by` merge into one theme. A merged theme's gold
then covers all its labels from the start.

The script prints every merge at startup for inspection:

```
[label-cluster] 'required_under' <- ['governed_by', 'regulated_by']
```

If merges look wrong (unrelated labels combined → raise the threshold; obvious synonyms not
merged → lower it), tune the constant and re-run.

---

## 6. Files touched

| file | change |
|---|---|
| `chatbot/data/expand_lightrag_gold.py` | **new** — expands gold on existing eval rows (this doc, §3) |
| `chatbot/data/generate_lightrag_eval.py` | label-similarity clustering for theme buckets; shared `embed_texts()` helper |
| `chatbot/data/.cache/relation_embeddings_v1.pkl` | **generated** — one-time cache of relation embeddings |
| `chatbot/data/eval_singlehop_lightrag_v2_expanded.jsonl` | **generated** — expanded eval set (original file untouched) |
| `chatbot/evaluate_lightrag.ipynb` | reranker now includes an evidence-chunk snippet in cross-encoder pairs; new batch-eval cells (`run_lightrag_pipeline`, `evaluate_lightrag_dataset`, recall/precision vs gold) |
