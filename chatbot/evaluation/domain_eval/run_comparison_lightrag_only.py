#!/usr/bin/env python3
"""LightRAG-only domain-filtering comparison: baseline (no domain filter, today's
LightRAG retrieval code unscoped) vs the new single|multi RRF -> verifier domain
filter -- both run through LightRAG (run_aggregation) ONLY, with Hybrid RAG
skipped entirely for both conditions.

Why exclude Hybrid RAG: it always searches the full index regardless of domain
scope (by design), so it behaves identically either way and would dilute the
signal in a with/without-domain-filtering comparison -- not a fair comparison
of the thing actually being tested.

Runs both conditions concurrently in one shared thread pool (like
run_comparison.py), scores each answer with the LLM judge, then reuses
_metrics.compare()/print_report() unchanged (this module produces result rows
in the exact same shape _common._run_one already produces).

    cd chatbot
    python scripts/domain_eval/run_comparison_lightrag_only.py \\
        [--max-workers 1] [--limit N] [--no-judge] [--judge-max-workers 4]

Writes:
  domain_data/eval_results/lightrag_only_single_multi.json
  domain_data/eval_results/lightrag_only_baseline.json
  domain_data/eval_results/lightrag_only_comparison_report.json
and prints a summary to stdout.
"""

import argparse
import json
import logging

import _lightrag_only
import _llm_judge
import _metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=1,
                        help="concurrent pipeline calls across BOTH conditions combined "
                             "(default 1 = sequential; the cross-encoder reranker in fusion.py "
                             "isn't safe to call from multiple threads at once on this machine "
                             "and will segfault under concurrency)")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--judge", dest="judge", action="store_true", default=True,
                        help="score each answer with the LLM judge (default: on)")
    parser.add_argument("--no-judge", dest="judge", action="store_false",
                        help="skip LLM-judge scoring")
    parser.add_argument("--judge-max-workers", type=int, default=1,
                        help="the judge is a plain LLM call (no reranker involved), so this "
                             "one is safe to raise even with --max-workers left at 1")
    args = parser.parse_args()

    questions = _lightrag_only.load_questions(limit=args.limit)
    filtered_results, baseline_results = _lightrag_only.run_both_conditions(
        questions, max_workers=args.max_workers)

    if args.judge:
        _llm_judge.judge_all(filtered_results, max_workers=args.judge_max_workers)
        _llm_judge.judge_all(baseline_results, max_workers=args.judge_max_workers)

    _lightrag_only.save_results(filtered_results, _lightrag_only.RESULTS_DIR / "lightrag_only_single_multi.json")
    _lightrag_only.save_results(baseline_results, _lightrag_only.RESULTS_DIR / "lightrag_only_baseline.json")

    # _metrics.compare(with_results, without_results) -- "with" = the filtered
    # condition, "without" = baseline, matching run_comparison.py's convention.
    report = _metrics.compare(filtered_results, baseline_results)
    _lightrag_only.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _lightrag_only.RESULTS_DIR / "lightrag_only_comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote comparison report -> {report_path}")

    _metrics.print_report(report)


if __name__ == "__main__":
    main()
