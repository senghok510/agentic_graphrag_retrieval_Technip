#!/usr/bin/env python3
"""Diagnostic: how many relations does each domain actually have classified
into it (CLASSIFIED_AS), vs how many documents are assigned to it at the
document level (HAS_LOW_LEVEL / HAS_MID_LEVEL / HAS_HIGH_DOMAIN)?

Why this matters: query-time domain prediction can only ever retrieve well if
the domain it predicts actually has enough classified relations to retrieve
FROM. A domain with near-zero CLASSIFIED_AS relations will produce a thin or
empty graph-scoped result even when the query->domain prediction itself is
"correct" -- that's a relation-classification coverage problem
(relation_classification.ipynb), not a query-prediction problem, and no
amount of prompt/RRF tuning on the query side fixes it.

Cross-references against domain_data/domain_eval_questions.json's
expected_domain labels so the domains we've actually been testing against
(and specifically the ones that keep showing up as misclassified -- see
domain_data/eval_results/result.md) are called out directly, sorted by
relation_count ascending so the sparsest domains surface first.

    cd chatbot
    python scripts/domain_eval/experiments/check_classified_as_coverage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CHATBOT_ROOT = Path(__file__).resolve().parents[3]
if str(CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATBOT_ROOT))

from src.pipeline.clients import neo4j_driver  # noqa: E402

QUESTIONS_PATH = CHATBOT_ROOT / "domain_data" / "domain_eval_questions.json"

_QUERY = """
MATCH (d)
WHERE d:LowLevelDomain OR d:MidLevelDomain OR d:HighLevelDomain
OPTIONAL MATCH (r:Relation)-[:CLASSIFIED_AS]->(d)
WITH d, count(DISTINCT r) AS relation_count
OPTIONAL MATCH (doc:Document)-[:HAS_LOW_LEVEL|HAS_MID_LEVEL|HAS_HIGH_DOMAIN]->(d)
WITH d, relation_count, count(DISTINCT doc) AS document_count
RETURN labels(d)       AS labels,
       d.domainName    AS domain_name,
       relation_count,
       document_count
ORDER BY relation_count ASC
"""

_TOTALS_QUERY = """
RETURN
    count { (r:Relation) }                    AS total_relations,
    count { ()-[:CLASSIFIED_AS]->() }          AS total_classified_as_edges,
    count { (r:Relation) WHERE NOT (r)-[:CLASSIFIED_AS]->() } AS relations_with_no_domain
"""


def load_expected_domains() -> set:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return {q["domain"] for q in data["questions"]}


def main() -> None:
    expected_domains = load_expected_domains()

    with neo4j_driver().session() as session:
        rows = [r.data() for r in session.run(_QUERY)]
        totals = session.run(_TOTALS_QUERY).single().data()

    print("=" * 88)
    print("Relation-classification coverage per domain")
    print("=" * 88)
    print(f"Total relations in graph:            {totals['total_relations']}")
    print(f"Total CLASSIFIED_AS edges:            {totals['total_classified_as_edges']}")
    print(f"Relations with NO domain at all:      {totals['relations_with_no_domain']}")
    print()
    print(f"{'domain_name':<38}{'tier':<14}{'relation_count':>15}{'document_count':>15}  in_eval_set")
    print("-" * 100)
    for row in rows:
        labels = [l for l in row["labels"] if l != "Domain"]
        tier = labels[0] if labels else "?"
        name = row["domain_name"] or "?"
        flag = "  <-- TESTED" if name in expected_domains else ""
        print(f"{name:<38}{tier:<14}{row['relation_count']:>15}{row['document_count']:>15}{flag}")

    print()
    tested_rows = [r for r in rows if r["domain_name"] in expected_domains]
    if tested_rows:
        sparse = [r for r in tested_rows if r["relation_count"] == 0]
        thin = [r for r in tested_rows if 0 < r["relation_count"] < 10]
        print(f"Tested domains with ZERO classified relations ({len(sparse)}): "
              f"{[r['domain_name'] for r in sparse]}")
        print(f"Tested domains with FEWER THAN 10 classified relations ({len(thin)}): "
              f"{[r['domain_name'] for r in thin]}")
    print("=" * 88)


if __name__ == "__main__":
    main()
