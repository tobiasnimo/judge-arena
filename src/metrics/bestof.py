"""
Metric: Best-of-Two Response Comparison
Task: Given a question and two responses (A and B), predict which one is better.
Score: Accuracy — fraction of times the judge picks the correct winner.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from leaderboard import update_metric_leaderboard, update_overall_leaderboard


ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT / "datasets" / "bestof.json"

PROMPT_TEMPLATE = """\
You are an impartial AI judge. Your task is to evaluate two responses to a question \
and determine which one is better.

**Question:**
{question}

**Response A:**
{response_a}

**Response B:**
{response_b}

Compare both responses based on accuracy, completeness, clarity, and helpfulness. \
Choose the better response, or "tie" if they are equally good.

Respond with JSON only, no explanation:
{{"winner": "A"}} or {{"winner": "B"}} or {{"winner": "tie"}}\
"""

# Map dataset winner labels to the judge's expected output tokens
_WINNER_MAP = {"model_a": "a", "model_b": "b", "tie": "tie"}


def run(judge) -> dict:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    correct = 0
    unparseable = 0

    pbar = tqdm(dataset, desc=f"bestof [{judge.name}]")
    for item in pbar:
        pbar.set_postfix(q=item["question"][:60])
        prompt = PROMPT_TEMPLATE.format(
            question=item["question"],
            response_a=item["model_a"],
            response_b=item["model_b"],
        )
        output = judge.generate(prompt)
        parsed = judge.parse_json(output)

        if parsed is None or "winner" not in parsed:
            unparseable += 1
            continue

        predicted = parsed["winner"].strip().lower()
        actual = _WINNER_MAP.get(item["winner"].strip().lower(), item["winner"].strip().lower())

        if predicted == actual:
            correct += 1

    total = len(dataset)
    evaluated = total - unparseable
    accuracy = round(correct / evaluated, 4) if evaluated > 0 else 0.0

    result = {
        "model_name": judge.name,
        "model_id": judge.model_id,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "unparseable": unparseable,
        "last_run": datetime.now(timezone.utc).isoformat(),
    }

    update_metric_leaderboard("bestof", result, sort_key="accuracy", ascending=False)
    update_overall_leaderboard(
        judge.model_id, judge.name, {"bestof_accuracy": accuracy}
    )

    return result
