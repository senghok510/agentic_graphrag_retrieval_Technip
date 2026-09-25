#!/usr/bin/env python3
"""Diagnostic: why do 1,629 relations have zero CLASSIFIED_AS edges?

Two hypotheses to distinguish:
  (a) They were classified "GENERAL" by relation_classification.ipynb, but
      there's no GENERAL LowLevelDomain/MidLevelDomain node for the write-back
      MATCH to land on -- the classification decision exists, it just has
      nowhere to write to.
  (b) The relation classifier is over-triggering "GENERAL" the same way the
      query-time gate we tried (and killed) did: its candidate domain list is
      bounded by its OWN document's already-assigned domains
      (GRAPH_CONSTRUCTION.md Sec 5) -- if the document itself has no
      HAS_LOW_LEVEL/HAS_MID_LEVEL domains, every relation inside it is
      starved of real candidates and GENERAL becomes the only/easy answer,
      even when a specific domain would have fit had it been offered.

This script checks (1) whether a GENERAL domain node exists at any tier, (2)
what fraction of the 1,629 orphan relations come from documents that
themselves have zero low/mid-level domains assigned (directly testing
hypothesis b across the FULL population, not just a sample), and (3) prints a
random text sample of orphan relations + their source document's assigned
domains, for a human sanity check on whether "GENERAL" actually looks right.

    cd chatbot
    python scripts/domain_eval/experiments/check_general_classification_gap.py [--sample 15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHATBOT_ROOT = Path(__file__).resolve().parents[3]
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from src.pipeline.clients import neo4j_driver  # noqa: E402

_GENERAL_NODE_QUERY = """
MATCH (d)
WHERE (d:LowLevelDomain OR d:MidLevelDomain OR d:HighLevelDomain) AND d.domainName = 'GENERAL'
RETURN labels(d) AS labels, d.domainName AS domain_name
"""

_CANDIDATE_STARVATION_QUERY = """
MATCH (r:Relation)
WHERE NOT (r)-[:CLASSIFIED_AS]->()
MATCH (d:Document)<-[:PART_OF]-(:Chunk)-[:ASSERTS]->(r)
OPTIONAL MATCH (d)-[:HAS_LOW_LEVEL]->(low:LowLevelDomain)
OPTIONAL MATCH (d)-[:HAS_MID_LEVEL]->(mid:MidLevelDomain)
WITH r, count(DISTINCT low) AS n_low, count(DISTINCT mid) AS n_mid
RETURN
    count(r)                                          AS total_orphan_relations,
    sum(CASE WHEN n_low = 0 AND n_mid = 0 THEN 1 ELSE 0 END) AS orphans_with_zero_doc_domains,
    sum(CASE WHEN n_low > 0 OR n_mid > 0 THEN 1 ELSE 0 END)  AS orphans_with_doc_domains_available
"""

_SAMPLE_QUERY = """
MATCH (r:Relation)
WHERE NOT (r)-[:CLASSIFIED_AS]->()
MATCH (d:Document)<-[:PART_OF]-(:Chunk)-[:ASSERTS]->(r)
OPTIONAL MATCH (subj:Entity)-[:SUBJECT_OF]->(r)
OPTIONAL MATCH (r)-[:OBJECT_OF]->(obj:Entity)
OPTIONAL MATCH (d)-[:HAS_LOW_LEVEL]->(low:LowLevelDomain)
OPTIONAL MATCH (d)-[:HAS_MID_LEVEL]->(mid:MidLevelDomain)
WITH r, d, subj, obj,
     collect(DISTINCT low.domainName) AS doc_low_domains,
     collect(DISTINCT mid.domainName) AS doc_mid_domains
RETURN r.relationId AS relation_id,
       r.relationshipLabel AS label,
       r.relationshipDescription AS description,
       subj.name AS subject,
       obj.name AS object,
       d.fileName AS filename,
       doc_low_domains,
       doc_mid_domains
ORDER BY rand()
LIMIT $sample_size
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=15, help="how many orphan relations to print for eyeballing")
    args = parser.parse_args()

    with neo4j_driver().session() as session:
        general_nodes = [r.data() for r in session.run(_GENERAL_NODE_QUERY)]
        starvation = session.run(_CANDIDATE_STARVATION_QUERY).single().data()
        sample = [r.data() for r in session.run(_SAMPLE_QUERY, {"sample_size": args.sample})]

    print("=" * 88)
    print("1. Does a GENERAL domain node exist at any tier?")
    print("=" * 88)
    if general_nodes:
        for row in general_nodes:
            labels = [l for l in row["labels"] if l != "Domain"]
            print(f"  FOUND: {row['domain_name']!r} as {labels}")
        print("  -> A GENERAL node exists. The write-back MATCH should be able to land --")
        print("     if orphan relations still exist, check whether their DOCUMENT has a")
        print("     HAS_LOW_LEVEL/HAS_MID_LEVEL edge to this exact node (see section 2).")
    else:
        print("  NONE FOUND. No LowLevelDomain/MidLevelDomain/HighLevelDomain node named")
        print("  'GENERAL' exists in this graph -- hypothesis (a) confirmed: any relation")
        print("  classified GENERAL has nowhere to write its CLASSIFIED_AS edge to, full stop.")

    print()
    print("=" * 88)
    print("2. Of the orphan relations, how many come from documents with NO low/mid domain at all?")
    print("=" * 88)
    total = starvation["total_orphan_relations"]
    zero_doc = starvation["orphans_with_zero_doc_domains"]
    have_doc = starvation["orphans_with_doc_domains_available"]
    pct = round(100 * zero_doc / total, 1) if total else 0.0
    print(f"  Total orphan relations (no CLASSIFIED_AS edge): {total}")
    print(f"  ...from documents with ZERO HAS_LOW_LEVEL/HAS_MID_LEVEL domains: {zero_doc} ({pct}%)")
    print(f"  ...from documents that DO have low/mid domains assigned:         {have_doc}")
    print()
    if total and zero_doc / total > 0.5:
        print("  -> Majority are candidate-starved (hypothesis b): their own document had no")
        print("     domain to offer the relation classifier, so GENERAL wasn't really a choice.")
    elif total:
        print("  -> Majority have documents WITH real domain candidates available, yet the")
        print("     relation still ended up GENERAL/unclassified -- points more toward the")
        print("     classifier itself over-using GENERAL rather than candidate starvation.")

    print()
    print("=" * 88)
    print(f"3. Random sample of {len(sample)} orphan relations (eyeball: does GENERAL look right?)")
    print("=" * 88)
    for row in sample:
        print(f"- [{row['relation_id'][:8]}] {row.get('subject','?')} --{row.get('label','?')}--> {row.get('object','?')}")
        print(f"    {row.get('description','')}")
        print(f"    file: {row.get('filename','?')}")
        print(f"    doc's low domains: {row.get('doc_low_domains') or '(none)'} | "
              f"mid domains: {row.get('doc_mid_domains') or '(none)'}")
        print()
    print("=" * 88)


if __name__ == "__main__":
    main()
