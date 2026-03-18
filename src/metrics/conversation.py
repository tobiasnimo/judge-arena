"""
Metric: Answer Quality Scoring
Task: Given a question, a generated answer, and a ground truth answer,
      assign a score (0–1) reflecting how closely the generated answer matches the ground truth.
Score: MAE — mean absolute error between judge scores and dataset scores.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from leaderboard import update_metric_leaderboard, update_overall_leaderboard


ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT / "datasets" / "conversational.json"

PROMPT_TEMPLATE = """\
You are an impartial AI judge. Your task is to evaluate how accurately a generated \
answer matches a ground truth answer.

**Question:**
{question}

**Generated Answer:**
{gen_answer}

**Ground Truth Answer:**
{gt_answer}

Score the generated answer from 0.0 to 1.0 based on how closely it matches the ground truth:
- 1.0: Perfect match or semantically equivalent
- 0.75: Mostly correct with minor omissions or inaccuracies
- 0.5: Partially correct, captures the main idea but misses key details
- 0.25: Mostly incorrect with limited relevant content
- 0.0: Completely wrong, irrelevant, or contradicts the ground truth

Respond with JSON only, no explanation:
{{"score": <float between 0.0 and 1.0>}}\
"""


def run(judge) -> dict:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    errors = []
    unparseable = 0

    pbar = tqdm(dataset, desc=f"conversation [{judge.name}]")
    for item in pbar:
        pbar.set_postfix(q=item["question"][:60])
        prompt = PROMPT_TEMPLATE.format(
            question=item["question"],
            gen_answer=item["gen_answer"],
            gt_answer=item["gt_answer"],
        )
        output = judge.generate(prompt)
        parsed = judge.parse_json(output)

        if parsed is None or "score" not in parsed:
            unparseable += 1
            continue

        try:
            predicted = float(parsed["score"])
            predicted = max(0.0, min(1.0, predicted))  # clamp to [0, 1]
        except (ValueError, TypeError):
            unparseable += 1
            continue

        actual = float(item["score"])
        errors.append(abs(predicted - actual))

    total = len(dataset)
    mae = round(sum(errors) / len(errors), 4) if errors else None

    result = {
        "model_name": judge.name,
        "model_id": judge.model_id,
        "mae": mae,
        "total": total,
        "unparseable": unparseable,
        "last_run": datetime.now(timezone.utc).isoformat(),
    }

    update_metric_leaderboard("conversation", result, sort_key="mae", ascending=True)
    update_overall_leaderboard(
        judge.model_id, judge.name, {"conversation_mae": mae}
    )

    return result
