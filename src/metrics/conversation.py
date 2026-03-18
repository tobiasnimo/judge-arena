"""
Metric: Answer Quality Scoring
Task: Given a question, a generated answer, and a ground truth answer,
      assign a score (0–1) reflecting how closely the generated answer matches the ground truth.
Score: MAE — mean absolute error between judge scores and dataset scores.
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.base import ScoreOutput
from leaderboard import save_metric_results, update_leaderboard


ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT / "datasets" / "conversational.json"

PROMPT_TEMPLATE = """You are an impartial AI judge. Your task is to evaluate how accurately a generated answer matches a ground truth answer.

**Question:**
{question}

**Generated Answer:**
{gen_answer}

**Ground Truth Answer:**
{gt_answer}

Score the generated answer from 0.0 to 1.0 based on how closely it matches the ground truth:
- 1.0: Perfect match or semantically equivalent
- 0.0: Completely wrong, irrelevant, or contradicts the ground truth

Respond with JSON only,:
```json
{{
"reasoning": "<one sentence explaining your score>",
"score": <float between 0.0 and 1.0>
}}
```"""



def run(judge, debug: bool = False) -> dict:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    errors = []
    unparseable = 0
    rows = []

    pbar = tqdm(dataset, desc=f"conversation [{judge.name}]")
    for item in pbar:
        pbar.set_postfix(q=item["question"][:60])
        prompt = PROMPT_TEMPLATE.format(
            question=item["question"],
            gen_answer=item["gen_answer"],
            gt_answer=item["gt_answer"],
        )
        output = judge.generate(prompt, schema=ScoreOutput)
        parsed = judge.parse_json(output)

        if parsed is None or "score" not in parsed:
            unparseable += 1
            row = {
                "question": item["question"],
                "actual_score": item["score"],
                "predicted_score": None,
                "reasoning": None,
                "error": None,
                "parseable": False,
            }
            if debug:
                row["raw"] = output
            rows.append(row)
            continue

        try:
            predicted = float(parsed["score"])
            predicted = max(0.0, min(1.0, predicted))
        except (ValueError, TypeError):
            unparseable += 1
            row = {
                "question": item["question"],
                "actual_score": item["score"],
                "predicted_score": None,
                "reasoning": None,
                "error": None,
                "parseable": False,
            }
            if debug:
                row["raw"] = output
            rows.append(row)
            continue

        actual = float(item["score"])
        error = abs(predicted - actual)
        errors.append(error)

        row = {
            "question": item["question"],
            "actual_score": actual,
            "predicted_score": predicted,
            "reasoning": parsed.get("reasoning"),
            "error": round(error, 4),
            "parseable": True,
        }
        if debug:
            row["raw"] = output
        rows.append(row)

    total = len(dataset)
    mae = round(sum(errors) / len(errors), 4) if errors else None

    result = {
        "model_name": judge.name,
        "model_id": judge.model_id,
        "mae": mae,
        "total": total,
        "unparseable": unparseable,
    }

    save_metric_results("conversation", judge.model_id, rows)
    update_leaderboard(judge.model_id, "conversation", mae, total, unparseable)

    return result
