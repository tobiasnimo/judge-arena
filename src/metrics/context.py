"""
Metric: Context Relevance Scoring
Task: Given a question and a context passage, assign a score (0–1) reflecting
      how helpful the context is for answering the question.
Score: MAE — mean absolute error between judge scores and dataset scores.
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from leaderboard import save_metric_results, update_leaderboard


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
{{"reasoning": "<one sentence explaining your score>", "score": <float between 0.0 and 1.0>}}\
"""


def run(judge) -> dict:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    errors = []
    unparseable = 0
    rows = []

    pbar = tqdm(dataset, desc=f"context [{judge.name}]")
    for item in pbar:
        pbar.set_postfix(q=item["question"][:60])
        prompt = PROMPT_TEMPLATE.format(
            question=item["question"],
            context=item["context"],
        )
        output = judge.generate(prompt)
        parsed = judge.parse_json(output)

        if parsed is None or "score" not in parsed:
            unparseable += 1
            rows.append({
                "question": item["question"],
                "actual_score": item["score"],
                "predicted_score": None,
                "reasoning": None,
                "error": None,
                "parseable": False,
            })
            continue

        try:
            predicted = float(parsed["score"])
            predicted = max(0.0, min(1.0, predicted))
        except (ValueError, TypeError):
            unparseable += 1
            rows.append({
                "question": item["question"],
                "actual_score": item["score"],
                "predicted_score": None,
                "reasoning": None,
                "error": None,
                "parseable": False,
            })
            continue

        actual = float(item["score"])
        error = abs(predicted - actual)
        errors.append(error)

        rows.append({
            "question": item["question"],
            "actual_score": actual,
            "predicted_score": predicted,
            "reasoning": parsed.get("reasoning"),
            "error": round(error, 4),
            "parseable": True,
        })

    total = len(dataset)
    mae = round(sum(errors) / len(errors), 4) if errors else None

    result = {
        "model_name": judge.name,
        "model_id": judge.model_id,
        "mae": mae,
        "total": total,
        "unparseable": unparseable,
    }

    save_metric_results("context", judge.model_id, rows)
    update_leaderboard(judge.model_id, "context", mae, total, unparseable)

    return result
