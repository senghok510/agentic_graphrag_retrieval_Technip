"""
Synthetic multi-hop eval data via subgraph sampling (PageRank / type_3).

Idea: every Relation is linked to the Chunk(s) that ASSERT it, so the graph gives the
retrieval ground truth for free. We sample a connected subgraph spanning >=2 distinct
chunks, ask an LLM for a multi-hop question + grounded answer, then filter (deterministic
+ LLM judge). Gold = the required relations/entities/chunks.

Run:  python chatbot/data/generate_subgraph_eval.py --n 10
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

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI

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

adj = defaultdict(list)
degree = Counter()
for r in state["all_relation_nodes"]:
    s, o = r["subject_key"], r["object_key"]
    if s not in entity_by_key or o not in entity_by_key:
        continue
    adj[s].append((r["relation_id"], o))
    adj[o].append((r["relation_id"], s))
    degree[s] += 1
    degree[o] += 1

HUB_THRESHOLD = 150
hub_keys = {k for k, d in degree.items() if d >= HUB_THRESHOLD}


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


def relation_view(rel_id: str) -> dict:
    r = relation_by_id[rel_id]
    s, o = entity_by_key[r["subject_key"]], entity_by_key[r["object_key"]]
    asserts = assertions_by_relation.get(rel_id, [])
    return {
        "relation_id": rel_id,
        "subject": s["name"], "object": o["name"],
        "label": r.get("relationship_label"),
        "description": r.get("relationship_description"),
        "evidence": [
            {"chunk_id": a["chunk_id"], "file_name": a.get("file_name"),
             "page": a.get("page_number"), "text": (a.get("evidence_text") or "").strip(),
             "confidence": a.get("confidence")}
            for a in asserts
        ],
    }


@dataclass
class Subgraph:
    relation_ids: list
    entity_keys: list
    chunk_ids: list
    sampler: str
    via_hub: bool

    def n_hops(self): return len(self.relation_ids)
    def n_distinct_chunks(self): return n_distinct_chunks(self.chunk_ids)
    def spans_files(self):
        files = set()
        for rid in self.relation_ids:
            for a in assertions_by_relation.get(rid, []):
                files.add(a.get("file_name"))
        return len(files)


def relation_endpoints(rel_id):
    r = relation_by_id[rel_id]
    return r["subject_key"], r["object_key"]


def join_nodes(rel_ids):
    cnt = Counter()
    for rid in rel_ids:
        for k in relation_endpoints(rid):
            cnt[k] += 1
    return {k for k, c in cnt.items() if c >= 2}


def _finalize(rel_ids, sampler):
    rel_ids = list(dict.fromkeys(rel_ids))
    ent_keys, chunk_ids = set(), set()
    for rid in rel_ids:
        s, o = relation_endpoints(rid)
        ent_keys.update([s, o])
        chunk_ids |= relation_chunks(rid)
    return Subgraph(rel_ids, sorted(ent_keys), sorted(chunk_ids), sampler, bool(ent_keys & hub_keys))


# --------------------------------------------------------------------------------------
# samplers
# --------------------------------------------------------------------------------------
def sample_path(n_relations=2, max_tries=200):
    non_hub_seeds = [k for k in adj if k not in hub_keys and degree[k] >= 1]
    for _ in range(max_tries):
        cur = random.choice(non_hub_seeds)
        path, visited = [], {cur}
        ok = True
        for _ in range(n_relations):
            choices = [(rid, nb) for rid, nb in adj[cur] if rid not in path and nb not in visited]
            if not choices:
                ok = False
                break
            rid, nb = random.choice(choices)
            path.append(rid)
            visited.add(nb)
            cur = nb
        if ok and len(path) == n_relations:
            return _finalize(path, "path")
    return None


def sample_multiseed(n_relations=3, max_tries=200):
    seeds = [k for k in adj if k not in hub_keys and 2 <= degree[k] <= 40]
    if not seeds:
        seeds = [k for k in adj if k not in hub_keys]
    for _ in range(max_tries):
        seed = random.choice(seeds)
        frontier = {seed}
        rel_ids = []
        for _ in range(n_relations * 4):
            if len(rel_ids) >= n_relations:
                break
            node = random.choice(list(frontier))
            choices = [(rid, nb) for rid, nb in adj[node] if rid not in rel_ids]
            if not choices:
                continue
            rid, nb = random.choice(choices)
            rel_ids.append(rid)
            frontier.add(nb)
        if len(rel_ids) >= n_relations:
            return _finalize(rel_ids[:n_relations], "multiseed")
    return None


# --------------------------------------------------------------------------------------
# prefilters
# --------------------------------------------------------------------------------------
STOP_CONNECTOR_NAMES = {
    "project", "the project", "contract", "the contract", "works", "the works",
    "company group", "tender", "contractor", "company", "tenderer",
}
stop_connector_keys = {
    k for k, e in entity_by_key.items()
    if (e.get("name") or "").strip().lower() in STOP_CONNECTOR_NAMES
}
stop_connector_keys |= hub_keys


def passes_prefilters(sg, *, min_distinct_chunks=2, max_hub_nodes=1, min_confidence=0.5,
                      block_stop_connectors=True):
    if sg is None:
        return False
    if sg.n_distinct_chunks() < min_distinct_chunks:
        return False
    if len(set(sg.entity_keys) & hub_keys) > max_hub_nodes:
        return False
    if block_stop_connectors and (join_nodes(sg.relation_ids) & stop_connector_keys):
        return False
    for rid in sg.relation_ids:
        confs = [a.get("confidence", 1.0) for a in assertions_by_relation.get(rid, [])]
        if confs and max(confs) < min_confidence:
            return False
    return True


# --------------------------------------------------------------------------------------
# LLM generation + judge
# --------------------------------------------------------------------------------------
GEN_SYS = (
    "You generate high-quality MULTI-HOP evaluation questions for a tender/EPC (ITB) "
    "Graph-RAG system. A multi-hop question can only be answered by combining facts from "
    "MORE THAN ONE source chunk. You must ground every claim strictly in the provided evidence."
)


def render_subgraph_for_llm(sg) -> str:
    lines = []
    for i, rid in enumerate(sg.relation_ids, 1):
        v = relation_view(rid)
        lines.append(f"RELATION R{i}: {v['subject']} --[{v['label']}]--> {v['object']}")
        if v["description"]:
            lines.append(f"  meaning: {v['description']}")
        for a in v["evidence"]:
            lines.append(f"  evidence (chunk={a['chunk_id']} | {a['file_name']} p.{a['page']}): {a['text'][:600]}")
    return "\n".join(lines)


def generate_qa(sg):
    user = f"""You are given a connected subgraph of a tender knowledge graph. Each RELATION is
backed by evidence from a specific source chunk.

{render_subgraph_for_llm(sg)}

Write ONE natural question that a tender analyst might ask, such that:
- answering it REQUIRES combining at least TWO relations that come from DIFFERENT chunks,
- it is NOT answerable from any single chunk alone,
- it does not leak the answer, and is specific (names the real entities, not "R1"/"R2"),
- PREFER a 'bridge' question (the answer requires inferring an intermediate link) over a
  'conjunction' (two unrelated facts about the same entity glued with "and").

Then give the expected answer, grounded ONLY in the evidence above. The answer must resolve
the SPECIFIC connection asked - not merely assert that both endpoints relate to a common node.

Classify the question:
- "bridge"      : needs an intermediate entity/relation to connect the endpoints.
- "conjunction" : two parallel facts about the same named entity.
- "thematic"    : aggregates/summarizes across several relations.

Return STRICT JSON:
{{
  "question": "...",
  "answer": "...",
  "question_type": "bridge|conjunction|thematic",
  "required_relations": ["R1","R2"],
  "reasoning": "one line: why it needs multiple chunks",
  "self_multihop": true
}}"""
    out = call_llm_json(GEN_SYS, user)
    tag_to_rid = {f"R{i}": rid for i, rid in enumerate(sg.relation_ids, 1)}
    req = [tag_to_rid[t] for t in out.get("required_relations", []) if t in tag_to_rid]
    out["required_relation_ids"] = req or sg.relation_ids
    out["required_chunk_ids"] = sorted({c for rid in out["required_relation_ids"]
                                        for c in relation_chunks(rid)})
    return out


JUDGE_SYS = (
    "You are a strict validator of synthetic MULTI-HOP QA used to evaluate a tender Graph-RAG "
    "system. Use ONLY the provided evidence."
)


def judge_qa(sg, qa):
    user = f"""EVIDENCE (grouped by relation/chunk):
{render_subgraph_for_llm(sg)}

QUESTION: {qa['question']}
ANSWER:   {qa['answer']}

Score each 0-5 and decide acceptance. A good item is faithful, genuinely multi-hop
(needs >=2 chunks), specific, non-trivial, and the answer must RESOLVE the connection asked.
REJECT tautologies: if the question already states/contains the answer, set answer_independence=0.

Return STRICT JSON:
{{
  "faithfulness": 0-5,
  "multihop_necessity": 0-5,
  "answer_completeness": 0-5,
  "answer_independence": 0-5,  // 0 if the QUESTION already states/contains the answer (tautology)
  "clarity": 0-5,
  "accept": true/false,
  "reason": "one line"
}}"""
    return call_llm_json(JUDGE_SYS, user)


def deterministic_ok(sg, qa, min_required_chunks=2):
    return n_distinct_chunks(qa.get("required_chunk_ids", [])) >= min_required_chunks


def entity_named_in(text: str, ent_key: str) -> bool:
    """True if the entity's name (or an alias) appears verbatim in the text."""
    t = (text or "").lower()
    e = entity_by_key.get(ent_key, {})
    cands = [(e.get("name") or "").lower()] + [str(a).lower() for a in (e.get("aliases") or [])]
    return any(c and len(c) >= 3 and c in t for c in cands)


def hidden_connector_ok(qa) -> bool:
    """A real bridge must DISCOVER its connector: at least one join node of the required
    relations must NOT be named in the question. If every connector is already named, the
    question merely stitches given pieces (a conjunction in disguise)."""
    joins = join_nodes(qa["required_relation_ids"])
    if not joins:
        return False  # required relations share no entity -> not a connected multi-hop
    return any(not entity_named_in(qa["question"], k) for k in joins)


def required_entities(qa) -> set:
    ents = set()
    for rid in qa["required_relation_ids"]:
        ents.update(relation_endpoints(rid))
    return ents


def enough_entities(qa, min_entities=3) -> bool:
    """A genuine multi-hop chain/star needs >=3 distinct entities. Two relations over the
    SAME pair of entities (parallel edges) is not multi-hop -- it is one restated fact."""
    return len(required_entities(qa)) >= min_entities


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
def build_dataset(n_target=10, judge_threshold=4, max_attempts=None, verbose=True):
    samplers = [
        ("path", lambda: sample_path(n_relations=random.choice([2, 3]))),
        ("multiseed", lambda: sample_multiseed(n_relations=random.choice([3, 4]))),
    ]
    max_attempts = max_attempts or n_target * 12
    rows, attempts, seen = [], 0, set()
    rejects = Counter()

    while len(rows) < n_target and attempts < max_attempts:
        attempts += 1
        _, fn = random.choice(samplers)
        sg = fn()
        if not passes_prefilters(sg):
            rejects["prefilter"] += 1
            continue
        try:
            qa = generate_qa(sg)
        except Exception as e:
            rejects["gen_error"] += 1
            if verbose:
                print("  gen error:", str(e)[:120])
            continue
        if not qa.get("self_multihop", True) or not deterministic_ok(sg, qa):
            rejects["not_multihop"] += 1
            continue
        # a "bridge" must hide its connector; otherwise it's a conjunction in disguise
        if not hidden_connector_ok(qa):
            rejects["named_connector"] += 1
            continue
        # >=3 distinct entities -> a real chain/star, not a same-pair parallel edge
        if not enough_entities(qa):
            rejects["too_few_entities"] += 1
            continue
        q_norm = re.sub(r"\W+", " ", qa["question"].lower()).strip()
        if q_norm in seen:
            rejects["dup"] += 1
            continue
        try:
            verdict = judge_qa(sg, qa)
        except Exception as e:
            rejects["judge_error"] += 1
            if verbose:
                print("  judge error:", str(e)[:120])
            continue
        gate = [verdict.get("faithfulness", 0), verdict.get("multihop_necessity", 0),
                verdict.get("answer_completeness", 0), verdict.get("answer_independence", 0)]
        if not verdict.get("accept") or min(gate) < judge_threshold:
            rejects["judge_reject"] += 1
            continue

        req_sg = _finalize(qa["required_relation_ids"], sg.sampler)
        seen.add(q_norm)
        rows.append({
            "question": qa["question"],
            "reference_answer": qa["answer"],
            "question_type": qa.get("question_type"),
            "gold_chunk_ids": req_sg.chunk_ids,
            "gold_relation_ids": req_sg.relation_ids,
            "gold_entity_keys": req_sg.entity_keys,
            "gold_entity_names": [entity_by_key[k]["name"] for k in req_sg.entity_keys],
            "sampler": sg.sampler,
            "n_hops": req_sg.n_hops(),
            "n_distinct_chunks": req_sg.n_distinct_chunks(),
            "spans_files": req_sg.spans_files(),
            "via_hub": req_sg.via_hub,
            "sampled_relation_ids": sg.relation_ids,
            "sampled_chunk_ids": sg.chunk_ids,
            "judge": verdict,
            "gen_reasoning": qa.get("reasoning"),
        })
        if verbose:
            print(f"  [{len(rows)}/{n_target}] type={qa.get('question_type')} "
                  f"hops={req_sg.n_hops()} chunks={req_sg.n_distinct_chunks()} "
                  f"via_hub={req_sg.via_hub} | {qa['question'][:80]}")

    print(f"\nDONE: {len(rows)} rows in {attempts} attempts | rejects={dict(rejects)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--threshold", type=int, default=4)
    ap.add_argument("--out", default=str(repo_root / "chatbot/data/synthetic_qa_subgraph_multihop_v1.jsonl"))
    args = ap.parse_args()

    print(f"graph: entities={len(entity_by_key)} relations={len(relation_by_id)} "
          f"chunks={len(chunk_by_id)} hubs={[entity_by_key[k]['name'] for k in hub_keys]}")
    print(f"stop-connectors: {len(stop_connector_keys)} | target={args.n}\n")

    dataset = build_dataset(n_target=args.n, judge_threshold=args.threshold)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in dataset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(dataset)} rows -> {out}")

    if dataset:
        types = Counter(r["question_type"] for r in dataset)
        print(f"type mix: {dict(types)} | via_hub: {sum(r['via_hub'] for r in dataset)}/{len(dataset)}")


if __name__ == "__main__":
    main()
