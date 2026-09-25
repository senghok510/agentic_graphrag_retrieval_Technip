"""
Synthetic eval data for LightRAG (type_1) — dual-level: LOCAL + GLOBAL.

LightRAG retrieves two ways, so we generate two question shapes, each with gold derived
straight from the graph (the ASSERTS/neighborhood structure gives the ground truth for free):

  local  : ll-keywords -> ENTITY vector -> 1-hop neighborhood + chunks
           -> "what is X / describe X / X's key obligations"
           gold = seed entity, its required 1-hop relations, their chunks, ll-keywords(=name)

  global : hl-keywords -> RELATION vector -> relations + chunks
           -> "summarize / what are all the <theme> ..."
           gold = a theme set of relations, their chunks, hl-keywords(=theme)

Run:  python chatbot/data/generate_lightrag_eval.py --mode both --n 10
"""
import os
import json
import pickle
import re
import random
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

load_dotenv()
random.seed(13)


# --------------------------------------------------------------------------------------
# repo root + LLM
# --------------------------------------------------------------------------------------
def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "output" / "jupyter-notebook" / "checkpoints").exists():
            return p
    raise FileNotFoundError("Could not locate repo root")

repo_root = find_repo_root(Path(__file__).resolve())

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_APIM"),
    api_key=os.getenv("API_KEY", "DUMMY"),
    default_headers={"Ocp-Apim-Subscription-Key": os.getenv("APIM_SUBSCRIPTION_KEY")},
    temperature=0.4,
)


def parse_json_object(raw: str):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    return json.loads(m.group(0) if m else raw)


def call_llm_json(system: str, user: str, max_retries: int = 3):
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            return parse_json_object(resp.content)
        except Exception as e:
            last = e
            time.sleep(1.5 * attempt)
    raise last


embed_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_APIM"),
    api_key="DUMMY",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    default_headers={"Ocp-Apim-Subscription-Key": os.getenv("APIM_SUBSCRIPTION_KEY")},
)
EMBED_MODEL = "text-embedding-3-large-llm4t"


def embed_texts(texts, dimensions: int = 512, batch_size: int = 100, max_retries: int = 3):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        last = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = embed_client.embeddings.create(model=EMBED_MODEL, input=batch, dimensions=dimensions)
                out.extend(item.embedding for item in resp.data)
                last = None
                break
            except Exception as e:
                last = e
                time.sleep(1.5 * attempt)
        if last:
            raise last
    return out


# --------------------------------------------------------------------------------------
# load state + indexes
# --------------------------------------------------------------------------------------
STATE_FILE = repo_root / "output/jupyter-notebook/checkpoints/state_closed_v5_merge.pkl"
with open(STATE_FILE, "rb") as f:
    state = pickle.load(f)

entity_by_key = {e["entity_key"]: e for e in state["all_entity_nodes"] if e.get("entity_key")}
relation_by_id = {r["relation_id"]: r for r in state["all_relation_nodes"]}
chunk_by_id = {c["chunk_id"]: c for c in state["all_chunk_payloads"]}

assertions_by_relation = defaultdict(list)
for a in state["all_assertions"]:
    assertions_by_relation[a["relation_id"]].append(a)

adj = defaultdict(list)        # entity_key -> [(relation_id, neighbor_key)]
degree = Counter()
for r in state["all_relation_nodes"]:
    s, o = r["subject_key"], r["object_key"]
    if s not in entity_by_key or o not in entity_by_key:
        continue
    adj[s].append((r["relation_id"], o))
    adj[o].append((r["relation_id"], s))
    degree[s] += 1
    degree[o] += 1

HUB_THRESHOLD = 50
hub_keys = {k for k, d in degree.items() if d >= HUB_THRESHOLD}

# relations grouped by their (free-form) relationship_label = themes for global mode.
# Raw labels are extracted per-relation at graph-build time, so near-synonyms (e.g.
# "required_under" / "governed_by" / "must_comply_with") end up as separate exact-string
# buckets even though they express the same kind of fact. That understates recall for any
# retriever (LightRAG's included) that finds relations by embedding similarity rather than by
# this literal label — a "global" question built from one raw-label bucket will have gold that
# excludes equally-relevant relations sitting in a sibling bucket. To close that gap, raw labels
# are clustered by embedding similarity before grouping into themes.
LABEL_CLUSTER_SIM_THRESHOLD = 0.55  # tune by inspecting the "merged labels" printout below

rels_by_raw_label = defaultdict(list)
for rid, r in relation_by_id.items():
    lbl = (r.get("relationship_label") or "").strip()
    if lbl and {a["chunk_id"] for a in assertions_by_relation.get(rid, []) if a.get("chunk_id")}:
        rels_by_raw_label[lbl].append(rid)

_raw_labels = sorted(rels_by_raw_label, key=lambda l: -len(rels_by_raw_label[l]))  # biggest buckets seed clusters first
_label_vecs = dict(zip(_raw_labels, embed_texts([l.replace("_", " ") for l in _raw_labels]))) if _raw_labels else {}

_clusters = []  # [{"labels": [...], "centroid": np.array, "n": int}]
for lbl in _raw_labels:
    v = np.asarray(_label_vecs[lbl], dtype=np.float64)
    v = v / (np.linalg.norm(v) or 1.0)
    best_i, best_sim = -1, -1.0
    for i, c in enumerate(_clusters):
        sim = float(np.dot(v, c["centroid"]))
        if sim > best_sim:
            best_i, best_sim = i, sim
    if best_i >= 0 and best_sim >= LABEL_CLUSTER_SIM_THRESHOLD:
        c = _clusters[best_i]
        c["centroid"] = (c["centroid"] * c["n"] + v) / (c["n"] + 1)
        c["centroid"] /= np.linalg.norm(c["centroid"]) or 1.0
        c["n"] += 1
        c["labels"].append(lbl)
    else:
        _clusters.append({"labels": [lbl], "centroid": v, "n": 1})

rels_by_label = {}
for c in _clusters:
    rep = max(c["labels"], key=lambda l: len(rels_by_raw_label[l]))  # most common raw label names the theme
    rel_ids = list(dict.fromkeys(rid for l in c["labels"] for rid in rels_by_raw_label[l]))
    rels_by_label[rep] = rel_ids
    if len(c["labels"]) > 1:
        print(f"  [label-cluster] {rep!r} <- {[l for l in c['labels'] if l != rep]}")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def relation_chunks(rel_id: str) -> set:
    return {a["chunk_id"] for a in assertions_by_relation.get(rel_id, []) if a.get("chunk_id")}


def chunk_group(chunk_id: str) -> str:
    """Collapse adjacent chunk-splits of the same page into one source key."""
    return re.sub(r"_text_chunk_\d+$", "", chunk_id or "")


def n_distinct_chunks(chunk_ids) -> int:
    return len({chunk_group(c) for c in chunk_ids})


def relation_endpoints(rel_id):
    r = relation_by_id[rel_id]
    return r["subject_key"], r["object_key"]


def relation_view(rel_id: str) -> dict:
    r = relation_by_id[rel_id]
    s, o = entity_by_key[r["subject_key"]], entity_by_key[r["object_key"]]
    return {
        "relation_id": rel_id, "subject": s["name"], "object": o["name"],
        "label": r.get("relationship_label"), "description": r.get("relationship_description"),
        "evidence": [
            {"chunk_id": a["chunk_id"], "file_name": a.get("file_name"),
             "page": a.get("page_number"), "text": (a.get("evidence_text") or "").strip()}
            for a in assertions_by_relation.get(rel_id, [])
        ],
    }


def render_relations(rel_ids) -> str:
    lines = []
    for i, rid in enumerate(rel_ids, 1):
        v = relation_view(rid)
        lines.append(f"RELATION R{i}: {v['subject']} --[{v['label']}]--> {v['object']}")
        if v["description"]:
            lines.append(f"  meaning: {v['description']}")
        for a in v["evidence"]:
            lines.append(f"  evidence (chunk={a['chunk_id']} | {a['file_name']} p.{a['page']}): {a['text'][:500]}")
    return "\n".join(lines)


def finalize_gold(required_rel_ids):
    """From the LLM-required relations -> distinct chunks + endpoint entities."""
    rel_ids = list(dict.fromkeys(required_rel_ids))
    ent_keys, chunk_ids = set(), set()
    for rid in rel_ids:
        s, o = relation_endpoints(rid)
        ent_keys.update([s, o])
        chunk_ids |= relation_chunks(rid)
    return rel_ids, sorted(ent_keys), sorted(chunk_ids)


def map_required(out, rel_ids):
    """Map the LLM's R-tags back to relation ids (fallback: all)."""
    tag_to_rid = {f"R{i}": rid for i, rid in enumerate(rel_ids, 1)}
    req = [tag_to_rid[t] for t in out.get("required_relations", []) if t in tag_to_rid]
    return req or list(rel_ids)


# --------------------------------------------------------------------------------------
# samplers
# --------------------------------------------------------------------------------------
def sample_local_entity(min_deg=3, max_deg=40, max_tries=300):
    """A non-hub entity with a real neighborhood spanning >=2 distinct chunks."""
    cands = [k for k in adj if k not in hub_keys and min_deg <= degree[k] <= max_deg]
    for _ in range(max_tries):
        seed = random.choice(cands)
        rel_ids = list(dict.fromkeys(rid for rid, _ in adj[seed] if relation_chunks(rid)))
        chunks = {c for r in rel_ids for c in relation_chunks(r)}
        if len(rel_ids) >= 2 and n_distinct_chunks(chunks) >= 2:
            random.shuffle(rel_ids)
            return seed, rel_ids[:8]          # cap neighborhood shown to the LLM
    return None


def sample_theme(min_rels=3, max_rels=8, max_tries=300):
    """A relationship_label theme with >=min_rels relations spanning >=2 distinct chunks."""
    themes = [(lbl, rs) for lbl, rs in rels_by_label.items()
              if len(rs) >= min_rels
              and n_distinct_chunks({c for r in rs for c in relation_chunks(r)}) >= 2]
    if not themes:
        return None
    for _ in range(max_tries):
        lbl, rs = random.choice(themes)
        return lbl, random.sample(rs, min(max_rels, len(rs)))
    return None


# --------------------------------------------------------------------------------------
# LLM generation + judge
# --------------------------------------------------------------------------------------
GEN_SYS = ("You generate high-quality evaluation questions for a tender/EPC (ITB) Graph-RAG "
           "system (LightRAG). Ground every claim strictly in the provided evidence.")

HL_STYLE = ("schedule_dependency, payment_terms, compliance_obligation, liability_allocation, "
            "scope_inclusion, test_sequencing, approval_workflow, document_reference")


def generate_local(seed_key, rel_ids):
    e = entity_by_key[seed_key]
    user = f"""You are given a tender knowledge-graph ENTITY and its 1-hop neighborhood
(the relations touching it), each backed by evidence chunks.

ENTITY: {e['name']}
DESCRIPTION: {e.get('description', '')}

{render_relations(rel_ids)}

Write ONE natural question a tender analyst would ask to UNDERSTAND this entity
(what it is / its key obligations / requirements / role), answerable from the neighborhood.
- The question MUST be about this single entity and NAME it.
- It must require reading at least one relation (not answerable from the name alone).
Then give the expected answer, grounded ONLY in the evidence above.

Return STRICT JSON:
{{
  "question": "...",
  "answer": "...",
  "required_relations": ["R1","R2"],
  "question_type": "entity_overview",
  "self_ok": true
}}"""
    out = call_llm_json(GEN_SYS, user)
    out["required_relation_ids"] = map_required(out, rel_ids)
    return out


def generate_global(label, rel_ids):
    user = f"""You are given a THEME from a tender knowledge graph: a set of relations that share
the relationship type "{label}", each backed by evidence chunks.

{render_relations(rel_ids)}

Write ONE natural AGGREGATION / SUMMARY question a tender analyst would ask about this theme
across the project (e.g. "what are all the ... / summarize the ... / which ... apply").
- It must require COMBINING several relations, not a single fact.
- Be concrete about the theme; do not say "R1"/"R2".
Then give the expected answer, grounded ONLY in the evidence above.
Also extract 2-5 HIGH-LEVEL theme keywords (snake_case, lowercase; NO entity names or numbers),
in the same style as: {HL_STYLE}

Return STRICT JSON:
{{
  "question": "...",
  "answer": "...",
  "required_relations": ["R1","R2","R3"],
  "high_level_keywords": ["theme_one","theme_two"],
  "question_type": "thematic",
  "self_ok": true
}}"""
    out = call_llm_json(GEN_SYS, user)
    out["required_relation_ids"] = map_required(out, rel_ids)
    return out


JUDGE_SYS = ("You are a strict validator of synthetic QA used to evaluate a tender LightRAG "
             "system. Use ONLY the provided evidence.")


def judge(mode, rel_ids, qa):
    extra = ("single_entity_focus : 0-5, is the question clearly about ONE entity (the overview target)"
             if mode == "local" else
             "aggregation : 0-5, does it genuinely AGGREGATE across several relations (0 if a single fact)")
    user = f"""EVIDENCE:
{render_relations(rel_ids)}

QUESTION: {qa['question']}
ANSWER:   {qa['answer']}

Score each 0-5 and decide acceptance. Reject tautologies (question states the answer).

Return STRICT JSON:
{{
  "faithfulness": 0-5,
  "answer_completeness": 0-5,
  "answer_independence": 0-5,
  "{extra.split(':')[0].strip()}": 0-5,
  "clarity": 0-5,
  "accept": true/false,
  "reason": "one line"
}}
Where: {extra}"""
    return call_llm_json(JUDGE_SYS, user)


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
def make_local_row(judge_threshold):
    s = sample_local_entity()
    if not s:
        return None, "sample"
    seed_key, rel_ids = s
    qa = generate_local(seed_key, rel_ids)
    if not qa.get("self_ok", True):
        return None, "self"
    req, ent_keys, chunk_ids = finalize_gold(qa["required_relation_ids"])
    if not req or n_distinct_chunks(chunk_ids) < 1:
        return None, "thin"
    v = judge("local", rel_ids, qa)
    gate = [v.get("faithfulness", 0), v.get("answer_completeness", 0),
            v.get("answer_independence", 0), v.get("single_entity_focus", 0)]
    if not v.get("accept") or min(gate) < judge_threshold:
        return None, "judge"
    e = entity_by_key[seed_key]
    return {
        "mode": "local",
        "question": qa["question"], "reference_answer": qa["answer"],
        "question_type": qa.get("question_type", "entity_overview"),
        "gold_entity_key": seed_key, "gold_entity_name": e["name"],
        "gold_low_level_keywords": [e["name"]] + list(e.get("aliases", [])),
        "gold_relation_ids": req, "gold_chunk_ids": chunk_ids,
        "gold_entity_keys": ent_keys,
        "n_distinct_chunks": n_distinct_chunks(chunk_ids),
        "sampled_relation_ids": rel_ids, "judge": v,
    }, None


def make_global_row(judge_threshold):
    s = sample_theme()
    if not s:
        return None, "sample"
    label, rel_ids = s
    qa = generate_global(label, rel_ids)
    if not qa.get("self_ok", True):
        return None, "self"
    req, ent_keys, chunk_ids = finalize_gold(qa["required_relation_ids"])
    if len(req) < 2 or n_distinct_chunks(chunk_ids) < 2:
        return None, "thin"
    v = judge("global", rel_ids, qa)
    gate = [v.get("faithfulness", 0), v.get("answer_completeness", 0),
            v.get("answer_independence", 0), v.get("aggregation", 0)]
    if not v.get("accept") or min(gate) < judge_threshold:
        return None, "judge"
    return {
        "mode": "global",
        "question": qa["question"], "reference_answer": qa["answer"],
        "question_type": qa.get("question_type", "thematic"),
        "theme_label": label,
        "gold_high_level_keywords": qa.get("high_level_keywords", []),
        "gold_relation_ids": req, "gold_chunk_ids": chunk_ids,
        "gold_entity_keys": ent_keys,
        "n_distinct_chunks": n_distinct_chunks(chunk_ids),
        "sampled_relation_ids": rel_ids, "judge": v,
    }, None


def build_dataset(mode="both", n_target=10, judge_threshold=4, max_attempts=None, verbose=True):
    if mode == "both":
        plan = ["local", "global"] * ((n_target + 1) // 2)
        plan = plan[:n_target]
    else:
        plan = [mode] * n_target
    max_attempts = max_attempts or n_target * 12

    rows, attempts, seen, rejects = [], 0, set(), Counter()
    want = Counter(plan)
    got = Counter()

    while len(rows) < n_target and attempts < max_attempts:
        attempts += 1
        # pick the mode still under quota
        remaining = [m for m in ("local", "global") if got[m] < want[m]]
        if not remaining:
            break
        m = random.choice(remaining)
        try:
            row, why = (make_local_row if m == "local" else make_global_row)(judge_threshold)
        except Exception as e:
            rejects[f"{m}_error"] += 1
            if verbose:
                print(f"  {m} error:", str(e)[:120])
            continue
        if row is None:
            rejects[f"{m}_{why}"] += 1
            continue
        q_norm = re.sub(r"\W+", " ", row["question"].lower()).strip()
        if q_norm in seen:
            rejects["dup"] += 1
            continue
        seen.add(q_norm)
        got[m] += 1
        rows.append(row)
        if verbose:
            tag = (row.get("gold_entity_name") if m == "local" else row.get("theme_label"))
            print(f"  [{len(rows)}/{n_target}] {m:<6} chunks={row['n_distinct_chunks']} "
                  f"| {tag} | {row['question'][:70]}")

    print(f"\nDONE: {len(rows)} rows in {attempts} attempts | mix={dict(got)} | rejects={dict(rejects)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "global", "both"], default="both")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--threshold", type=int, default=4)
    ap.add_argument("--out", default=str(repo_root / "chatbot/data/synthetic_qa_lightrag_v2.jsonl"))
    args = ap.parse_args()

    print(f"graph: entities={len(entity_by_key)} relations={len(relation_by_id)} "
          f"chunks={len(chunk_by_id)} | themes(labels>=3 rels)="
          f"{sum(1 for rs in rels_by_label.values() if len(rs) >= 3)}")
    print(f"mode={args.mode} target={args.n}\n")

    dataset = build_dataset(mode=args.mode, n_target=args.n, judge_threshold=args.threshold)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in dataset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(dataset)} rows -> {out}")
    if dataset:
        print("mode mix:", dict(Counter(r["mode"] for r in dataset)))


if __name__ == "__main__":
    main()
