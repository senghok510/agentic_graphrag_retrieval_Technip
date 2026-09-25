"""Comparison metrics for the domain-filtering with/without eval runs.

Every metric here is computed purely from the pipeline's own telemetry
(``debug`` + the returned answer) — no gold answers needed, since the eval
questions only carry an *expected domain* label, not a gold answer text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from _llm_judge import ALL_SCORE_KEYS as JUDGE_KEYS


def _jaccard(a: List[str], b: List[str]) -> Optional[float]:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return None  # neither run cited anything -> not comparable
    return len(sa & sb) / len(sa | sb)


def _graph_chunk_count(result: Dict[str, Any]) -> Optional[int]:
    """candidate_counts[0] is the graph branch's chunk count for graph routes
    (orchestrator.py: candidate_lists = [graph_chunks, hybrid_chunks]); None for
    textual_factoid, which never touches the graph, so there's nothing to compare."""
    if not result.get("ok") or result.get("route") == "textual_factoid":
        return None
    counts = result.get("candidate_counts") or []
    return counts[0] if len(counts) > 1 else None


def compare(with_results: List[Dict[str, Any]], without_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_id_without = {r["id"]: r for r in without_results}
    per_question: List[Dict[str, Any]] = []

    scoped_correct = scoped_incorrect = full_scope = 0

    for w in with_results:
        wo = by_id_without.get(w["id"])
        if wo is None:
            continue

        expected = w["expected_domain"]
        predicted = w.get("predicted_domains") or []
        scope = w.get("domain_scope")

        if scope == "full":
            full_scope += 1
            domain_hit = None
        elif expected in predicted:
            scoped_correct += 1
            domain_hit = True
        else:
            scoped_incorrect += 1
            domain_hit = False

        row: Dict[str, Any] = {
            "id": w["id"],
            "expected_domain": expected,
            "route": w.get("route"),
            "domain_scope": scope,
            "predicted_domains": predicted,
            "domain_hit": domain_hit,
            "with_ok": w.get("ok"),
            "without_ok": wo.get("ok"),
        }

        if w.get("ok") and wo.get("ok"):
            row["with_confidence"] = w.get("confidence")
            row["without_confidence"] = wo.get("confidence")
            row["confidence_delta"] = round(w.get("confidence", 0) - wo.get("confidence", 0), 4)

            row["with_elapsed_s"] = w.get("elapsed_s")
            row["without_elapsed_s"] = wo.get("elapsed_s")
            row["elapsed_s_delta"] = round(w.get("elapsed_s", 0) - wo.get("elapsed_s", 0), 2)

            wg, wog = _graph_chunk_count(w), _graph_chunk_count(wo)
            row["with_graph_chunks"] = wg
            row["without_graph_chunks"] = wog
            row["graph_chunks_delta"] = (wg - wog) if wg is not None and wog is not None else None

            w_files = [r.get("file_name") for r in (w.get("references") or [])]
            wo_files = [r.get("file_name") for r in (wo.get("references") or [])]
            row["reference_jaccard"] = _jaccard(w_files, wo_files)

            row["answer_changed"] = (w.get("answer", "").strip() != wo.get("answer", "").strip())

            wj, woj = w.get("judge"), wo.get("judge")
            if wj and woj:
                row["with_judge"] = wj
                row["without_judge"] = woj
                row["judge_deltas"] = {
                    dim: round(wj[dim] - woj[dim], 2)
                    for dim in JUDGE_KEYS
                    if wj.get(dim) is not None and woj.get(dim) is not None
                }

        per_question.append(row)

    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in per_question if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _avg_judge(judge_key: str, dim: str) -> Optional[float]:
        vals = [r[judge_key][dim] for r in per_question
                if r.get(judge_key) and r[judge_key].get(dim) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    n_scoped = scoped_correct + scoped_incorrect
    n_answer_rows = sum(1 for r in per_question if "answer_changed" in r)
    n_changed = sum(1 for r in per_question if r.get("answer_changed"))

    with_judge_avg = {dim: _avg_judge("with_judge", dim) for dim in JUDGE_KEYS}
    without_judge_avg = {dim: _avg_judge("without_judge", dim) for dim in JUDGE_KEYS}
    judge_delta_avg = {
        dim: round(with_judge_avg[dim] - without_judge_avg[dim], 3)
        if with_judge_avg[dim] is not None and without_judge_avg[dim] is not None else None
        for dim in JUDGE_KEYS
    }

    return {
        "n_questions": len(per_question),
        "with_errors": sum(1 for r in with_results if not r.get("ok")),
        "without_errors": sum(1 for r in without_results if not r.get("ok")),
        "domain_prediction": {
            "scoped_correct": scoped_correct,
            "scoped_incorrect": scoped_incorrect,
            "full_scope_no_narrowing": full_scope,
            "accuracy_when_scoped": round(scoped_correct / n_scoped, 4) if n_scoped else None,
        },
        "deltas": {
            "avg_confidence_delta_with_minus_without": _avg("confidence_delta"),
            "avg_elapsed_s_delta_with_minus_without": _avg("elapsed_s_delta"),
            "avg_graph_chunks_delta_with_minus_without": _avg("graph_chunks_delta"),
            "avg_reference_jaccard": _avg("reference_jaccard"),
            "pct_answers_changed": round(n_changed / n_answer_rows, 4) if n_answer_rows else None,
        },
        "llm_judge": {
            "with_domain": with_judge_avg,
            "without_domain": without_judge_avg,
            "delta_with_minus_without": judge_delta_avg,
        },
        "per_question": per_question,
    }


def print_report(report: Dict[str, Any]) -> None:
    dp = report["domain_prediction"]
    dl = report["deltas"]
    pct_changed = dl["pct_answers_changed"]

    print()
    print("=" * 72)
    print(f"Domain filtering A/B report -- {report['n_questions']} questions")
    print("=" * 72)
    print(f"errors:  with={report['with_errors']}  without={report['without_errors']}")
    print()
    print("Domain prediction accuracy (against the eval set's expected_domain):")
    print(f"  scoped & correct:     {dp['scoped_correct']}")
    print(f"  scoped & WRONG:       {dp['scoped_incorrect']}")
    print(f"  predicted full graph: {dp['full_scope_no_narrowing']}  (no scoping attempted)")
    print(f"  accuracy when scoped: {dp['accuracy_when_scoped']}")
    print()
    print("With-filter vs without-filter deltas (with minus without):")
    print(f"  avg confidence delta:   {dl['avg_confidence_delta_with_minus_without']}")
    print(f"  avg elapsed_s delta:    {dl['avg_elapsed_s_delta_with_minus_without']}")
    print(f"  avg graph-chunk delta:  {dl['avg_graph_chunks_delta_with_minus_without']}")
    print(f"  avg reference overlap:  {dl['avg_reference_jaccard']}  (Jaccard, 1.0 = identical sources)")
    print(f"  % answers changed:      {pct_changed * 100:.1f}%" if pct_changed is not None else "  % answers changed:      n/a")
    print()

    lj = report.get("llm_judge")
    if lj and any(v is not None for v in lj["with_domain"].values()):
        print("LLM-as-judge scores (1-5, pointwise per answer, averaged):")
        print(f"  {'dimension':<14}{'with':>8}{'without':>10}{'delta':>10}")
        for dim in ("completeness", "relevance", "fluency", "accuracy", "overall"):
            w_val = lj["with_domain"].get(dim)
            wo_val = lj["without_domain"].get(dim)
            d_val = lj["delta_with_minus_without"].get(dim)
            print(f"  {dim:<14}{w_val!s:>8}{wo_val!s:>10}{d_val!s:>10}")
        print()

    misses = [r for r in report["per_question"] if r.get("domain_hit") is False]
    if misses:
        print(f"Questions where the predicted domain MISSED the expected one ({len(misses)}):")
        for r in sorted(misses, key=lambda r: r["id"]):
            print(f"  [{r['id']}] expected={r['expected_domain']!r} predicted={r['predicted_domains']}")
    print("=" * 72)
