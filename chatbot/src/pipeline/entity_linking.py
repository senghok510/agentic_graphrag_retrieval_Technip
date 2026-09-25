"""Stage: Online Entity Linking.

Mermaid node ``Online Entity Linking`` (shared by the Aggregation and Multi-hop
branches). Resolves the question's entity mentions to KG entity nodes via a
recall-broad → precision-strict cascade:

    entity_name_ft (BM25)         → candidate generation
    chunk-mediated (MENTIONS)     → candidate generation (semantic, name-free)
    lexical scoring               → rank
    conservative LLM MATCH filter → precision
    LLM dedupe                    → one representative per duplicate group

``link_entities`` is the light single-channel resolver used by the LightRAG
aggregation branch; ``fuse_ppr_seeds`` is the two-channel (full-text +
chunk-mediated) fusion that produces PPR seed entities for the multi-hop branch.
All logic is copied verbatim from ppr.py / evaluate_lightrag.py.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from .clients import call_json, neo4j_driver

logger = logging.getLogger("agent_flow.pipeline.entity_linking")

ENTITY_FT_TOPK = 100
CHUNK_ENTITY_TOPK_CHUNKS = 20
CHUNK_ENTITY_SEED_TOPK = 20

# ── Cypher ───────────────────────────────────────────────────────────────────

ENTITY_FT_CYPHER = """
CALL db.index.fulltext.queryNodes('entity_name_ft', $q)
YIELD node AS e, score AS ft_score

WITH e, ft_score
OPTIONAL MATCH (ch:Chunk)-[:MENTIONS]->(e)
WITH e, ft_score,
     [x IN collect(DISTINCT {
        chunk_id: ch.ChunkID,
        text: ch.text,
        file_name: ch.fileName,
        page_number: ch.pageNumber
     }) WHERE x.chunk_id IS NOT NULL][..2] AS support_chunks

OPTIONAL MATCH (e)-[:SUBJECT_OF]->(r_out:Relation)-[:OBJECT_OF]->(nbr_out:Entity)
WITH e, ft_score, support_chunks,
     collect(DISTINCT r_out.relationshipLabel) AS outgoing_relation_types,
     collect(DISTINCT nbr_out.name) AS outgoing_neighbors

OPTIONAL MATCH (nbr_in:Entity)-[:SUBJECT_OF]->(r_in:Relation)-[:OBJECT_OF]->(e)
WITH e, ft_score, support_chunks,
     outgoing_relation_types, outgoing_neighbors,
     collect(DISTINCT r_in.relationshipLabel) AS incoming_relation_types,
     collect(DISTINCT nbr_in.name) AS incoming_neighbors

WITH e, ft_score, support_chunks,
     [x IN outgoing_relation_types + incoming_relation_types WHERE x IS NOT NULL AND x <> ''][..8] AS relation_types,
     [x IN outgoing_neighbors + incoming_neighbors WHERE x IS NOT NULL][..8] AS connected_entities

RETURN e.entityKey AS entity_key,
       e.name AS entity_name,
       coalesce(e.normalizedName, '') AS normalized_name,
       coalesce(e.canonicalCategory, 'other') AS canonical_category,
       coalesce(e.llmCategory, '') AS llm_category,
       coalesce(e.description, '') AS description,
       coalesce(e.aliases, []) AS aliases,
       relation_types,
       connected_entities,
       support_chunks,
       ft_score
ORDER BY ft_score DESC
LIMIT $top
"""

VECTOR_ENTITY_EXPANSION_CYPHER = """
UNWIND $chunk_ids AS ck
MATCH (ch:Chunk {ChunkID: ck})-[:MENTIONS]->(e:Entity)

OPTIONAL MATCH (e)-[:SUBJECT_OF]->(r_out:Relation)-[:OBJECT_OF]->(nbr_out:Entity)
WITH ck, ch, e,
     collect(DISTINCT r_out.relationshipLabel) AS outgoing_relation_types,
     collect(DISTINCT nbr_out.name) AS outgoing_neighbors

OPTIONAL MATCH (nbr_in:Entity)-[:SUBJECT_OF]->(r_in:Relation)-[:OBJECT_OF]->(e)
WITH ck, ch, e, outgoing_relation_types, outgoing_neighbors,
     collect(DISTINCT r_in.relationshipLabel) AS incoming_relation_types,
     collect(DISTINCT nbr_in.name) AS incoming_neighbors

WITH ck, ch, e,
     [x IN outgoing_relation_types + incoming_relation_types WHERE x IS NOT NULL AND x <> ''][..8] AS relation_types,
     [x IN outgoing_neighbors + incoming_neighbors WHERE x IS NOT NULL][..8] AS connected_entities

RETURN ck AS chunk_id,
       ch.text AS text,
       ch.fileName AS file_name,
       ch.pageNumber AS page_number,
       e.entityKey AS entity_key,
       e.name AS entity_name,
       coalesce(e.normalizedName, '') AS normalized_name,
       coalesce(e.canonicalCategory, 'other') AS canonical_category,
       coalesce(e.llmCategory, '') AS llm_category,
       coalesce(e.description, '') AS description,
       coalesce(e.aliases, []) AS aliases,
       relation_types,
       connected_entities
"""

_LINK_STOP = {"the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "by", "from",
              "about", "requirement", "requirements", "related", "relevant"}


# ── Text helpers ─────────────────────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


# ── Lucene builders + lexical scoring (evaluate_lightrag.py) ─────────────────

def _entity_hint_variants(entity_hints: List[str]) -> Dict[str, List[str]]:
    variants = {}
    for hint in entity_hints:
        base = (hint or "").strip()
        if not base:
            continue
        norm = _norm(base)
        forms = {base, base.lower(), norm}
        for sep in ["-", "/", "_"]:
            forms.add(base.replace(sep, " "))
            forms.add(base.replace(sep, ""))
        toks = norm.split()
        if len(toks) > 1:
            forms.add(" ".join(toks))
            forms.add("".join(toks))
        cleaned, seen = [], set()
        for item in forms:
            inorm = _norm(item)
            if inorm and inorm not in seen:
                cleaned.append(item.strip())
                seen.add(inorm)
        variants[hint] = cleaned
    return variants


def _build_loose_lucene(entity_hints: List[str]) -> str:
    loose, variants = [], _entity_hint_variants(entity_hints)
    for hint in entity_hints:
        for form in variants.get(hint, [hint]):
            fnorm = _norm(form)
            toks = [t for t in fnorm.split() if len(t) >= 3 and t not in _LINK_STOP]
            for t in toks:
                loose.append(f"{t}~1" if len(t) >= 5 else t)
            if len(toks) >= 2:
                loose.append(f'"{" ".join(toks)}"~3')
    return " OR ".join(dict.fromkeys(loose))


def _score_entity_candidates(rows, entity_hints, category_hints=None, relation_hints=None):
    category_hints, relation_hints = set(category_hints or []), set(relation_hints or [])
    max_ft = max([float(r.get("ft_score") or 0.0) for r in rows] or [1.0])
    out = []
    for row in rows:
        names = [x for x in [row.get("entity_name"), row.get("normalized_name"),
                             row.get("llm_category"), *row.get("aliases", [])] if x]
        best_sim, matched = 0.0, []
        for hint in entity_hints:
            hn = _norm(hint)
            htoks = [t for t in hn.split() if len(t) >= 3]
            m = 0.0
            for cand in names:
                cn = _norm(cand)
                if not cn:
                    continue
                ctoks = [t for t in cn.split() if len(t) >= 3]
                overlap = len(set(htoks) & set(ctoks))
                ratio = overlap / max(1, len(set(htoks)))
                if hn == cn:
                    m = max(m, 1.0)
                elif hn in cn or cn in hn:
                    a, b = sorted((len(hn), len(cn)))
                    m = max(m, 0.90 + 0.08 * (a / max(1, b)))
                elif overlap > 0:
                    m = max(m, 0.65 * ratio + 0.35 * SequenceMatcher(None, hn, cn).ratio())
            if m >= 0.52:
                matched.append(hint)
            best_sim = max(best_sim, m)
        rel_types = [x for x in row.get("relation_types", []) if x]
        ft_norm = float(row.get("ft_score") or 0.0) / max_ft
        score = (0.70 * ft_norm + 0.20 * best_sim + 0.05 * len(matched)
                 + 0.03 * len(relation_hints & set(rel_types)))
        if row.get("canonical_category") in category_hints:
            score += 0.02
        out.append({**row, "matched_hints": matched,
                    "best_hint_similarity": best_sim, "local_score": score, "seed_score": score,
                    "source": "entity_fulltext", "source_scores": {"entity_fulltext": score}})
    out.sort(key=lambda x: x["local_score"], reverse=True)
    return out


# ── Conservative LLM MATCH filter + dedupe (ppr.py) ──────────────────────────

def format_candidate_for_prompt(candidate: Dict[str, Any], rank: int) -> Dict[str, Any]:
    compact_support = []
    for ch in (candidate.get("support_chunks", []) or [])[:2]:
        compact_support.append({
            "file_name": ch.get("file_name", "Unknown"),
            "page_number": ch.get("page_number", "N/A"),
            "text": truncate_text(ch.get("text", ch.get("content", "")), max_chars=350),
        })
    return {
        "rank": rank,
        "entity_key": str(candidate.get("entity_key", "")),
        "entity_name": candidate.get("entity_name"),
        "llm_category": candidate.get("llm_category", ""),
        "aliases": candidate.get("aliases", [])[:8],
        "description": truncate_text(candidate.get("description", ""), max_chars=450),
        "matched_hints": candidate.get("matched_hints", []),
        "support_chunks": compact_support,
    }


def build_batch_entity_linking_prompt(user_query, entity_hints, entity_ft_candidates,
                                      expected_categories=None, top_k=30) -> str:
    candidates = []
    for rank, candidate in enumerate(entity_ft_candidates[:top_k], start=1):
        candidates.append({
            "rank": rank,
            "entity_key": candidate.get("entity_key", ""),
            "entity_name": candidate.get("entity_name", ""),
            "canonical_category": candidate.get("canonical_category", ""),
            "llm_category": candidate.get("llm_category", ""),
            "aliases": candidate.get("aliases", [])[:8],
            "description": truncate_text(candidate.get("description", ""), 350),
            "matched_hints_from_retrieval": candidate.get("matched_hints", []),
            "support_chunks": [
                {"text": truncate_text(ch.get("text", ch.get("content", "")), 250)}
                for ch in (candidate.get("support_chunks", []) or [])[:2]
            ],
        })
    payload = {
        "user_query": user_query,
        "entity_hints": entity_hints,
        "expected_categories": expected_categories or [],
        "candidate_kg_nodes": candidates,
    }
    return f"""
You are an entity linker for a tendering and EPC knowledge graph.

Task:
For each extracted entity hint, decide which candidate KG node(s), if any, refer to the same real-world entity or same procurement/technical concept.

Important rules:

- Evaluate every candidate independently.
- Keep multiple candidates only when they represent distinct useful graph concepts, or when near-duplicate nodes may carry different graph relations.
- Return MATCH only for exact name, alias, identifier, document code, tender bulletin number, deliverable ID, or strongly equivalent concept.
- Do NOT match only because candidates are semantically related.
- Similar procurement terms may be legally different.
- A hint may match multiple candidates only if they are duplicates, aliases, or near-equivalent nodes.
- If no reliable match exists for a hint, return no match for that hint.
- Be conservative. Wrong KG links are worse than missing KG links.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return STRICT JSON only:
{{
  "matches": [
    {{
      "entity_hint": "copy exact hint",
      "candidate_entity_key": "copy exact candidate entity_key",
      "candidate_entity_name": "copy exact candidate entity_name",
      "decision": "MATCH",
      "confidence": 0.0,
      "reason": "brief reason"
    }}
  ],
  "non_matches": [
    {{
      "entity_hint": "copy exact hint",
      "reason": "brief reason if no candidates matched"
    }}
  ]
}}

Confidence guide:
- 0.90-1.00: exact name/alias/code/identifier match
- 0.75-0.89: strong equivalent wording with consistent context
- below 0.75: usually do not match
""".strip()


def batch_link_entities_with_llm(user_query, entity_hints, entity_ft_candidates,
                                 expected_categories=None, top_k=30, confidence_threshold=0.80):
    """Conservative LLM MATCH/NON-MATCH filter over lexical candidates."""
    prompt = build_batch_entity_linking_prompt(
        user_query=user_query, entity_hints=entity_hints,
        entity_ft_candidates=entity_ft_candidates,
        expected_categories=expected_categories, top_k=top_k)
    data = call_json(
        system_prompt=("You are a conservative entity linker for a tendering and EPC "
                       "knowledge graph. Return strict JSON only."),
        user_prompt=prompt,
    )
    raw_matches = data.get("matches", []) or []
    candidate_by_key = {str(c.get("entity_key")): c for c in entity_ft_candidates if c.get("entity_key")}

    filtered, seen = [], set()
    for match in raw_matches:
        if str(match.get("decision", "")).upper() != "MATCH":
            continue
        if float(match.get("confidence") or 0.0) < confidence_threshold:
            continue
        candidate_key = str(match.get("candidate_entity_key") or "")
        candidate = candidate_by_key.get(candidate_key)
        if not candidate or candidate_key in seen:
            continue
        seen.add(candidate_key)
        candidate = dict(candidate)
        candidate["llm_entity_link_confidence"] = float(match.get("confidence") or 0.0)
        candidate["llm_entity_link_reason"] = match.get("reason", "")
        candidate["llm_matched_hint"] = match.get("entity_hint", "")
        filtered.append(candidate)
    return filtered, raw_matches


def build_seed_dedup_prompt(user_query, matched_entities) -> str:
    compact_entities = []
    for rank, ent in enumerate(matched_entities, start=1):
        compact_entities.append({
            "rank": rank,
            "entity_key": ent.get("entity_key") or ent.get("candidate_entity_key", ""),
            "entity_name": ent.get("entity_name") or ent.get("candidate_entity_name", ""),
            "canonical_category": ent.get("canonical_category") or ent.get("candidate_category", ""),
            "llm_category": ent.get("llm_category", ""),
            "aliases": ent.get("aliases", [])[:8],
            "description": truncate_text(ent.get("description", ""), 300),
            "matched_hint": ent.get("llm_matched_hint") or ent.get("entity_hint", ""),
            "confidence": ent.get("confidence") or ent.get("llm_entity_link_confidence", 0.0),
        })
    payload = {"user_query": user_query, "matched_entities": compact_entities}
    return f"""
You are cleaning a matched entity seed list for Personalized PageRank retrieval in an EPC/tender knowledge graph.

Task:
Group duplicate, alias, acronym-only, singular/plural, and near-equivalent entities.
Choose one preferred representative per duplicate group.
Keep distinct useful graph concepts as separate seeds.

Important rules:
- Do NOT remove entities merely because they are related.
- Remove or suppress only entities that are duplicate/alias/near-equivalent variants of another entity.
- Prefer the more descriptive/full entity name over acronym-only nodes.
- Prefer the entity with clearer technical meaning and richer name.
- Singular/plural variants are usually duplicates.
- Closely related but distinct concepts should both be kept.
- For relational queries, preserve seeds from both sides of the relationship.
- If unsure whether two entities are duplicates, keep both.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return STRICT JSON only:
{{
  "kept_entities": [
    {{
      "entity_key": "copy exact entity_key of preferred representative",
      "entity_name": "copy exact entity_name of preferred representative",
      "group_id": "short stable group label",
      "group_members": [{{"entity_key": "copy exact entity_key", "entity_name": "copy exact entity_name"}}],
      "reason": "brief reason why this representative is kept"
    }}
  ],
  "removed_entities": [
    {{
      "entity_key": "copy exact removed entity_key",
      "entity_name": "copy exact removed entity_name",
      "merged_into_entity_key": "copy exact kept representative entity_key",
      "merged_into_entity_name": "copy exact kept representative entity_name",
      "reason": "brief reason"
    }}
  ]
}}
""".strip()


def dedupe_matched_seed_entities_with_llm(user_query, matched_entities) -> Dict[str, Any]:
    prompt = build_seed_dedup_prompt(user_query=user_query, matched_entities=matched_entities)
    data = call_json(
        system_prompt=("You deduplicate matched KG entity seeds for graph retrieval. "
                       "Return strict JSON only."),
        user_prompt=prompt,
    )
    kept = data.get("kept_entities", []) or []
    removed = data.get("removed_entities", []) or []

    by_key = {}
    for ent in matched_entities:
        key = str(ent.get("entity_key") or ent.get("candidate_entity_key") or "")
        if key:
            by_key[key] = ent
    kept_keys = {str(row.get("entity_key")) for row in kept if row.get("entity_key")}
    removed_keys = {str(row.get("entity_key")) for row in removed if row.get("entity_key")}

    final_entities = []
    for kept_row in kept:
        key = str(kept_row.get("entity_key") or "")
        original = by_key.get(key)
        if not original:
            continue
        enriched = dict(original)
        enriched["dedupe_group_id"] = kept_row.get("group_id", "")
        enriched["dedupe_group_members"] = kept_row.get("group_members", [])
        enriched["dedupe_reason"] = kept_row.get("reason", "")
        enriched["is_deduped_representative"] = True
        final_entities.append(enriched)

    for key, original in by_key.items():
        if key not in kept_keys and key not in removed_keys:
            enriched = dict(original)
            enriched["dedupe_group_id"] = ""
            enriched["dedupe_group_members"] = [{
                "entity_key": key,
                "entity_name": original.get("entity_name") or original.get("candidate_entity_name", ""),
            }]
            enriched["dedupe_reason"] = "Missing from LLM dedupe output; kept defensively."
            enriched["is_deduped_representative"] = True
            final_entities.append(enriched)

    return {"final_entities": final_entities, "kept_entities": kept,
            "removed_entities": removed, "raw": data}


# ── Public resolvers ─────────────────────────────────────────────────────────

def link_entities(question, entity_hints=None, category_hints=None, relation_hints=None,
                  ft_topk=50, confidence_threshold=0.80) -> List[Dict[str, Any]]:
    """Single-channel resolver (LightRAG aggregation branch).

    full-text → lexical score → conservative LLM filter. Returns the filtered
    KG entity candidates (each carries ``entity_key``).
    """
    entity_hints = [h for h in (entity_hints or []) if h and h.strip()]
    if not entity_hints:
        return []
    loose = _build_loose_lucene(entity_hints)
    if not loose:
        return []
    with neo4j_driver().session() as s:
        rows = [r.data() for r in s.run(ENTITY_FT_CYPHER, {"q": loose, "top": ft_topk})]
    if not rows:
        return []
    candidates = _score_entity_candidates(rows, entity_hints, category_hints, relation_hints)
    filtered, _ = batch_link_entities_with_llm(
        user_query=question, entity_hints=entity_hints, entity_ft_candidates=candidates,
        expected_categories=list(category_hints or []), top_k=50,
        confidence_threshold=confidence_threshold)
    return filtered


def _entity_ft_candidates(entity_hints, category_hints, relation_hints, top=ENTITY_FT_TOPK):
    """Channel 1: BM25 over entity_name_ft → lexically scored candidates."""
    loose = _build_loose_lucene(entity_hints)
    if not loose:
        return []
    with neo4j_driver().session() as s:
        rows = [r.data() for r in s.run(ENTITY_FT_CYPHER, {"q": loose, "top": top})]
    return _score_entity_candidates(rows, entity_hints, category_hints, relation_hints)


def _chunk_mediated_candidates(entity_hints, vector_rows, top_chunks=CHUNK_ENTITY_TOPK_CHUNKS,
                               seed_topk=CHUNK_ENTITY_SEED_TOPK):
    """Channel 2: top vector chunks → MENTIONS entities → hint-scored candidates.

    Name-free recall: finds relevant chunks by meaning, reads off the entities
    they mention. Verbatim port of ppr.py's chunk-entity seed cell.
    """
    chosen = vector_rows[:top_chunks]
    chunk_ids = [row.get("chunk_id") for row in chosen if row.get("chunk_id")]
    if not chunk_ids:
        return []
    with neo4j_driver().session() as s:
        rows_raw = [r.data() for r in s.run(VECTOR_ENTITY_EXPANSION_CYPHER, {"chunk_ids": chunk_ids})]

    candidates_by_key = {}
    for row in rows_raw:
        entity_key = row.get("entity_key")
        if not entity_key or not row.get("chunk_id"):
            continue
        best_sim = 0.0
        matched = []
        names = [x for x in [row.get("entity_name"), row.get("normalized_name"),
                             row.get("llm_category"), *row.get("aliases", [])] if x]
        for hint in entity_hints:
            hn = _norm(hint)
            if not hn:
                continue
            m = 0.0
            for cand in names:
                cn = _norm(cand)
                if not cn:
                    continue
                if hn == cn:
                    m = max(m, 1.0)
                elif hn in cn or cn in hn:
                    a, b = sorted((len(hn), len(cn)))
                    m = max(m, 0.88 + 0.08 * (a / max(1, b)))
                else:
                    overlap = len(set(hn.split()) & set(cn.split())) / max(1, len(set(hn.split())))
                    seq = SequenceMatcher(None, hn, cn).ratio() * 0.9
                    m = max(m, overlap, seq)
            if m >= 0.30:
                matched.append(hint)
            best_sim = max(best_sim, m)

        support_chunk = {
            "chunk_id": row.get("chunk_id"), "text": row.get("text", ""),
            "file_name": row.get("file_name", "Unknown"), "page_number": row.get("page_number", "N/A"),
        }
        existing = candidates_by_key.get(entity_key)
        if existing is None:
            candidates_by_key[entity_key] = {
                "entity_key": entity_key, "entity_name": row.get("entity_name", ""),
                "canonical_category": row.get("canonical_category", "other"),
                "llm_category": row.get("llm_category", ""), "description": row.get("description", ""),
                "aliases": list(row.get("aliases", [])), "entity_matched_hints": matched,
                "connected_entities": list(row.get("connected_entities", [])),
                "support_chunks": [support_chunk], "source": "entity_from_top_chunks",
                "local_score": best_sim, "seed_score": best_sim,
                "source_scores": {"entity_from_top_chunks": best_sim},
            }
        else:
            existing["local_score"] += best_sim
            existing["seed_score"] = existing["local_score"]
            existing["source_scores"]["entity_from_top_chunks"] = (
                existing["source_scores"].get("entity_from_top_chunks", 0.0) + best_sim)
            existing["entity_matched_hints"] = list(dict.fromkeys(existing["entity_matched_hints"] + matched))
            if support_chunk["chunk_id"] not in {c.get("chunk_id") for c in existing["support_chunks"]}:
                existing["support_chunks"].append(support_chunk)

    return sorted(candidates_by_key.values(), key=lambda x: x["local_score"], reverse=True)[:seed_topk]


def fuse_ppr_seeds(analysis: dict, vector_rows: List[Dict[str, Any]], top: int = 50) -> List[Dict[str, Any]]:
    """Two-channel entity linking → RRF-fused PPR seed entities (multi-hop branch).

    Channel 1 (entity full-text) + Channel 2 (chunk-mediated) are fused by
    reciprocal-rank fusion, then confirmed by the conservative LLM MATCH filter
    and dedupe. Returns the final seed entities (each carries ``entity_key``).
    """
    entity_hints = [h.strip() for h in analysis.get("entity_hints", []) if h and h.strip()]
    category_hints = list(analysis.get("category_hints", []))
    relation_hints = list(analysis.get("relation_hints", []))
    if not entity_hints:
        return []

    ft_candidates = _entity_ft_candidates(entity_hints, category_hints, relation_hints)
    chunk_candidates = _chunk_mediated_candidates(entity_hints, vector_rows)

    # RRF fusion across the two channels (verbatim from ppr.py seed-fusion cell).
    fused_by_key = {}
    for source_name, candidates in [("entity_fulltext", ft_candidates),
                                    ("entity_from_top_chunks", chunk_candidates)]:
        ranked = sorted(candidates, key=lambda x: float(x.get("local_score") or x.get("seed_score") or 0.0), reverse=True)
        scores = [float(r.get("local_score") or r.get("seed_score") or 0.0) for r in ranked]
        score_min = min(scores) if scores else 0.0
        score_max = max(scores) if scores else 1.0
        for rank, row in enumerate(ranked, start=1):
            entity_key = row.get("entity_key")
            if not entity_key:
                continue
            raw_score = float(row.get("local_score") or row.get("seed_score") or 0.0)
            norm_score = ((raw_score - score_min) / (score_max - score_min)) if score_max > score_min else 1.0
            rec = fused_by_key.setdefault(entity_key, {
                "entity_key": entity_key, "entity_name": row.get("entity_name", ""),
                "canonical_category": row.get("canonical_category", "other"),
                "llm_category": row.get("llm_category", ""), "description": row.get("description", ""),
                "aliases": [], "entity_matched_hints": [], "relation_types": [], "support_chunks": [],
                "source_scores": defaultdict(float), "sources": [], "rrf_score": 0.0, "max_norm_score": 0.0,
            })
            rec["rrf_score"] += 1.0 / (60 + rank)
            rec["max_norm_score"] = max(rec["max_norm_score"], norm_score)
            rec["source_scores"][source_name] += norm_score
            rec["sources"] = list(dict.fromkeys(rec["sources"] + [source_name]))
            rec["aliases"] = list(dict.fromkeys(rec["aliases"] + list(row.get("aliases", []))))
            rec["entity_matched_hints"] = list(dict.fromkeys(
                rec["entity_matched_hints"] + list(row.get("entity_matched_hints", row.get("matched_hints", [])))))
            existing_ids = {c.get("chunk_id") for c in rec["support_chunks"]}
            for chunk in row.get("support_chunks", []):
                if chunk.get("chunk_id") not in existing_ids:
                    rec["support_chunks"].append(chunk)
                    existing_ids.add(chunk.get("chunk_id"))

    fused = []
    for rec in fused_by_key.values():
        rec["source_scores"] = dict(rec["source_scores"])
        rec["seed_score"] = rec["rrf_score"]
        rec["source"] = "+".join(rec["sources"])
        fused.append(rec)
    fused = sorted(fused, key=lambda x: x["seed_score"], reverse=True)[:top]

    if not fused:
        return []

    # Conservative confirmation + dedupe.
    filtered, _ = batch_link_entities_with_llm(
        user_query=analysis["question"], entity_hints=entity_hints, entity_ft_candidates=fused,
        expected_categories=category_hints, top_k=50, confidence_threshold=0.80)
    if not filtered:
        filtered = fused  # keep RRF seeds if the LLM filter is too strict / unavailable
    deduped = dedupe_matched_seed_entities_with_llm(user_query=analysis["question"], matched_entities=filtered)
    return deduped["final_entities"]
