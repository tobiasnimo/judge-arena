"""
CLI entry point for running LLM judge evaluations.

Usage:
    python run_judge.py --judge qwen-2b --metric bestof
    python run_judge.py --judge qwen-9b --metric all
    python run_judge.py --judge qwen-4b --metric conversation --backend transformers
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference import load_judge, list_judges
from metrics import bestof, conversation, context


METRICS = {
    "bestof": bestof,
    "conversation": conversation,
    "context": context,
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM as a judge across one or more metrics."
    )
    parser.add_argument(
        "--judge",
        required=True,
        choices=list_judges(),
        help="Judge model to evaluate.",
    )
    parser.add_argument(
        "--metric",
        default="all",
        choices=list(METRICS.keys()) + ["all"],
        help="Metric to run. Use 'all' to run every metric (default).",
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=["vllm", "transformers"],
        help="Inference backend (default: vllm).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Store raw LLM output in results JSON for debugging.",
    )
    args = parser.parse_args()

    judge = load_judge(args.judge, backend=args.backend)

    metrics_to_run = METRICS if args.metric == "all" else {args.metric: METRICS[args.metric]}

    all_results = {}
    for name, module in metrics_to_run.items():
        print(f"[INFO] Running metric: {name}")
        result = module.run(judge, debug=args.debug)
        all_results[name] = result
        print(f"[OUT]\n{json.dumps(result, indent=2)}")

    print(f"\n\n[INFO] Results written to: results/")


if __name__ == "__main__":
    main()
