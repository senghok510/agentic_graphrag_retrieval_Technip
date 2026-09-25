"""
Expand `global`-mode gold_relation_ids in an existing LightRAG eval set to close the
label-bucketing gap: generate_lightrag_eval.py builds each global "theme" by exact string match
on relationship_label (rels_by_label), so a relation asserting the same kind of fact under a
different label (e.g. governed_by vs required_under) is never gold — even though LightRAG's
actual retrieval (embedding similarity, label-agnostic) could legitimately surface it and get
penalized as a false positive.

For each global row: embed its hl-keywords, rank ALL evidenced relations in the graph by cosine
similarity, and ask an LLM judge whether each top candidate (not already gold) actually supports
the reference answer. Judged-relevant relations are merged into gold_relation_ids/gold_chunk_ids/
gold_entity_keys. `local` rows are passed through unchanged — their gold is a real 1-hop
neighborhood of one seed entity, not a label bucket, so this gap doesn't apply to them.

Run:  python chatbot/data/expand_lightrag_gold.py \
          --in chatbot/data/eval_singlehop_lightrag_v2.jsonl \
          --out chatbot/data/eval_singlehop_lightrag_v2_expanded.jsonl \
          --limit 3        # smoke-test a few rows before running the full set
"""
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_lightrag_eval import (          # reuse state + LLM/embedding plumbing already loaded there
    repo_root, entity_by_key, relation_by_id, relation_chunks, relation_view,
    relation_endpoints, n_distinct_chunks, call_llm_json, embed_texts,
)

EMBED_DIM = 512   # cheap dims — only used to rank candidates for the LLM judge, not the final gate


# --------------------------------------------------------------------------------------
# cache: embed every evidenced relation once (subject --label--> object. description)
# --------------------------------------------------------------------------------------
CACHE_FILE = repo_root / "chatbot/data/.cache/relation_embeddings_v1.pkl"


def relation_embed_text(rid):
    r = relation_by_id[rid]
    s, o = entity_by_key[r["subject_key"]], entity_by_key[r["object_key"]]
    return f"{s['name']} {r.get('relationship_label', '')} {o['name']}. {r.get('relationship_description', '')}"


def load_or_build_relation_embeddings():
    evidenced_ids = sorted(rid for rid in relation_by_id if relation_chunks(rid))
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        if set(cache["ids"]) == set(evidenced_ids):
            return cache["ids"], np.asarray(cache["vecs"], dtype=np.float32)
    print(f"embedding {len(evidenced_ids)} relations (one-time, cached to {CACHE_FILE}) ...")
    texts = [relation_embed_text(rid) for rid in evidenced_ids]
    vecs = np.asarray(embed_texts(texts), dtype=np.float32)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"ids": evidenced_ids, "vecs": vecs}, f)
    return evidenced_ids, vecs


# --------------------------------------------------------------------------------------
# judge: does this extra relation actually belong in gold for this question/answer?
# --------------------------------------------------------------------------------------
EXPAND_JUDGE_SYS = ("You are a strict validator building the gold answer-set for a tender/EPC "
                     "Graph-RAG evaluation. Use ONLY the evidence given per candidate.")


def judge_candidates(question, reference_answer, rel_ids):
    if not rel_ids:
        return []
    lines = []
    for i, rid in enumerate(rel_ids, 1):
        v = relation_view(rid)
        lines.append(f"CANDIDATE C{i} (relation_id={rid}): {v['subject']} --[{v['label']}]--> {v['object']}")
        if v["description"]:
            lines.append(f"  meaning: {v['description']}")
        for a in v["evidence"][:2]:
            lines.append(f"  evidence: {a['text'][:400]}")

    user = f"""QUESTION: {question}
REFERENCE ANSWER: {reference_answer}

Each CANDIDATE below is a graph relation that was NOT part of the original gold set (it was
extracted with a different relationship label but might still be relevant). For each candidate,
decide whether its fact would appear as ONE OF THE ENUMERATED ITEMS if the reference answer
were rewritten to be exhaustive.

ACCEPT only if the candidate states the same KIND of fact as the answer's items (e.g. if the
answer lists "X is required under act Y" facts, accept only other "thing required under / governed
by / regulated by a named act" facts).
REJECT candidates that are merely on the same topic: background obligations, process descriptions,
compliance duties, or anything that would be supporting context rather than a listed item.
When unsure, REJECT — a false gold entry is worse than a missing one.

{chr(10).join(lines)}

Return STRICT JSON: {{"relevant": ["C1","C3"]}}"""
    out = call_llm_json(EXPAND_JUDGE_SYS, user)
    tag_to_rid = {f"C{i}": rid for i, rid in enumerate(rel_ids, 1)}
    return [tag_to_rid[t] for t in out.get("relevant", []) if t in tag_to_rid]


# --------------------------------------------------------------------------------------
def expand_row(row, rel_ids, rel_vecs_normed, top_n=20, sim_floor=0.35):
    hl = row.get("gold_high_level_keywords") or []
    query_text = ", ".join(hl) if hl else row["question"]
    qvec = np.asarray(embed_texts([query_text])[0], dtype=np.float32)
    qvec = qvec / (np.linalg.norm(qvec) or 1.0)
    sims = rel_vecs_normed @ qvec

    existing = set(row.get("gold_relation_ids", []))
    order = np.argsort(-sims)
    candidates = []
    for idx in order:
        rid = rel_ids[idx]
        if rid in existing:
            continue
        if sims[idx] < sim_floor:
            break
        candidates.append(rid)
        if len(candidates) >= top_n:
            break

    added = judge_candidates(row["question"], row["reference_answer"], candidates)
    if not added:
        row["gold_expansion"] = {"candidates_considered": len(candidates),
                                 "candidate_relation_ids": candidates, "added_relation_ids": []}
        return row

    ent_keys, chunk_ids = set(row.get("gold_entity_keys", [])), set(row.get("gold_chunk_ids", []))
    for rid in added:
        s, o = relation_endpoints(rid)
        ent_keys.update([s, o])
        chunk_ids |= relation_chunks(rid)

    row["gold_relation_ids"] = sorted(existing | set(added))
    row["gold_entity_keys"] = sorted(ent_keys)
    row["gold_chunk_ids"] = sorted(chunk_ids)
    row["n_distinct_chunks"] = n_distinct_chunks(chunk_ids)
    row["gold_expansion"] = {"candidates_considered": len(candidates),
                             "candidate_relation_ids": candidates, "added_relation_ids": added}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--top-n", type=int, default=20, help="max new candidates sent to the LLM judge per row")
    ap.add_argument("--sim-floor", type=float, default=0.35, help="min cosine similarity to be considered a candidate")
    ap.add_argument("--limit", type=int, default=None, help="only expand the first N rows (rest pass through unchanged)")
    args = ap.parse_args()

    rel_ids, rel_vecs = load_or_build_relation_embeddings()
    rel_vecs_normed = rel_vecs / (np.linalg.norm(rel_vecs, axis=1, keepdims=True) + 1e-9)

    rows = [json.loads(l) for l in Path(args.inp).open(encoding="utf-8")]
    n_global = sum(1 for r in rows if r.get("mode") == "global")
    print(f"loaded {len(rows)} rows ({n_global} global) from {args.inp}")

    out_rows, n_expanded, n_added_total, n_done = [], 0, 0, 0
    for i, row in enumerate(rows, 1):
        if row.get("mode") != "global" or (args.limit and n_done >= args.limit):
            out_rows.append(row)
            continue
        n_done += 1
        before = len(row.get("gold_relation_ids", []))
        try:
            row = expand_row(row, rel_ids, rel_vecs_normed, top_n=args.top_n, sim_floor=args.sim_floor)
        except Exception as e:
            print(f"  [{i}] ERROR: {str(e)[:150]}")
            out_rows.append(row)
            continue
        after = len(row["gold_relation_ids"])
        if after > before:
            n_expanded += 1
            n_added_total += after - before
            print(f"  [{i}] +{after - before} relations | {row['question'][:70]}")
        out_rows.append(row)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDONE: expanded {n_expanded}/{n_done} checked global rows, "
          f"+{n_added_total} relations total -> {args.out}")


if __name__ == "__main__":
    main()
