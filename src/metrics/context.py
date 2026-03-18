"""
Metric: Context Relevance Scoring
Task: Given a question and a context passage, assign a score (0–1) reflecting
      how helpful the context is for answering the question.
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
DATASET_PATH = ROOT / "datasets" / "context.json"

PROMPT_TEMPLATE = """\
You are an impartial AI judge. Your task is to evaluate how helpful a context passage \
is for answering a given question.

**Question:**
{question}

**Context:**
{context}

Score the helpfulness of the context from 0.0 to 1.0:
- 1.0: Context directly and fully answers the question (Excellent)
- 0.6: Context is helpful and relevant but incomplete (Acceptable)
- 0.3: Context is marginally relevant, provides limited help (Could be Improved)
- 0.0: Context is not helpful, irrelevant, or misleading (Bad)

Respond with JSON only, no explanation:
{{"score": <float between 0.0 and 1.0>}}\
"""


def run(judge) -> dict:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    errors = []
    unparseable = 0

    for item in tqdm(dataset, desc=f"context [{judge.name}]"):
        prompt = PROMPT_TEMPLATE.format(
            question=item["question"],
            context=item["context"],
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

    update_metric_leaderboard("context", result, sort_key="mae", ascending=True)
    update_overall_leaderboard(
        judge.model_id, judge.name, {"context_mae": mae}
    )

    return result
