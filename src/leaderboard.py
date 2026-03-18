"""Shared leaderboard and results persistence used by all metrics."""

import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def save_metric_results(metric: str, model_id: str, rows: list):
    """Write per-item judge responses to results/[metric]/[model_id].json."""
    out_dir = RESULTS_DIR / metric
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_id.replace('/', '--')}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def update_leaderboard(model_id: str, updates: dict):
    """Upsert metric scores for a model into results/leaderboard.json.

    updates keys: bestof_accuracy, conversation_mae, context_mae
    overall_score is recomputed from whatever metrics are present.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "leaderboard.json"

    leaderboard = {}
    if path.exists():
        with open(path) as f:
            leaderboard = json.load(f)

    entry = leaderboard.setdefault(model_id, {})
    entry.update(updates)

    components = []
    if entry.get("bestof_accuracy") is not None:
        components.append(entry["bestof_accuracy"])
    if entry.get("conversation_mae") is not None:
        components.append(1 - entry["conversation_mae"])
    if entry.get("context_mae") is not None:
        components.append(1 - entry["context_mae"])

    entry["overall_score"] = round(sum(components) / len(components), 4) if components else None

    with open(path, "w") as f:
        json.dump(leaderboard, f, indent=2)
