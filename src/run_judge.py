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
from leaderboard import update_overall_leaderboard


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
    args = parser.parse_args()

    judge = load_judge(args.judge, backend=args.backend)

    metrics_to_run = METRICS if args.metric == "all" else {args.metric: METRICS[args.metric]}

    all_results = {}
    for name, module in metrics_to_run.items():
        print(f"\n{'='*50}")
        print(f"Running metric: {name}")
        print(f"{'='*50}")
        result = module.run(judge)
        all_results[name] = result
        print(f"Result: {json.dumps(result, indent=2)}")

    # When all three metrics are run together, refresh the overall leaderboard entry
    # (each metric already upserts its own column; this ensures overall_score is current)
    if args.metric == "all":
        update_overall_leaderboard(
            model_id=judge.model_id,
            model_name=judge.name,
            updates={
                "bestof_accuracy": all_results["bestof"]["accuracy"],
                "conversation_mae": all_results["conversation"]["mae"],
                "context_mae": all_results["context"]["mae"],
            },
        )
        print(f"\nOverall leaderboard updated.")

    print(f"\nDone. Leaderboard files written to: leaderboard/")


if __name__ == "__main__":
    main()
