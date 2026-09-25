#!/usr/bin/env python3
"""Run the domain eval set under BOTH filtering conditions concurrently (one
shared thread pool, so 'with' and 'without' pipeline calls are genuinely
interleaved rather than run back-to-back), then score the difference.

    cd chatbot
    python scripts/domain_eval/run_comparison.py [--max-workers 4] [--tender-id ROC_INPEX] [--limit N] [--no-judge]

LLM-as-judge scoring (completeness/relevance/fluency/accuracy, 1-5 each, + a
one-line reasoning per answer) runs by default -- pass --no-judge to skip it
for a faster/cheaper smoke run.

Writes:
  domain_data/eval_results/with_domain.json
  domain_data/eval_results/without_domain.json
  domain_data/eval_results/comparison_report.json
and prints a summary to stdout.
"""

import argparse
import json
import logging

import _common
import _llm_judge
import _metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=1,
                        help="concurrent pipeline calls across BOTH conditions combined "
                             "(default 1 = sequential; the cross-encoder reranker in fusion.py "
                             "isn't safe to call from multiple threads at once on this machine "
                             "and will segfault under concurrency -- raise this only if you've "
                             "confirmed that's not an issue for you)")
    parser.add_argument("--tender-id", default=None)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--judge", dest="judge", action="store_true", default=True,
                        help="score each answer with the LLM judge (default: on)")
    parser.add_argument("--no-judge", dest="judge", action="store_false",
                        help="skip LLM-judge scoring")
    parser.add_argument("--judge-max-workers", type=int, default=1,
                        help="the judge is a plain LLM call (no reranker involved), so this "
                             "one is safe to raise even with --max-workers left at 1")
    args = parser.parse_args()

    questions = _common.load_questions(limit=args.limit)
    with_results, without_results = _common.run_both_conditions(
        questions, tender_id=args.tender_id, max_workers=args.max_workers)

    if args.judge:
        _llm_judge.judge_all(with_results, max_workers=args.judge_max_workers)
        _llm_judge.judge_all(without_results, max_workers=args.judge_max_workers)

    _common.save_results(with_results, _common.RESULTS_DIR / "with_domain.json")
    _common.save_results(without_results, _common.RESULTS_DIR / "without_domain.json")

    report = _metrics.compare(with_results, without_results)
    _common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _common.RESULTS_DIR / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote comparison report -> {report_path}")

    _metrics.print_report(report)


if __name__ == "__main__":
    main()
