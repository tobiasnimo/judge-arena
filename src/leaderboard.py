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


def update_leaderboard(model_id: str, metric: str, score, total: int, failed: int):
    """Upsert a metric result for a model into results/leaderboard.json.

    Leaderboard structure:
    {
      "<model_id>": {
        "<metric>": {"score": ..., "total": ..., "failed": ...},
        ...
      },
      ...
    }
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "leaderboard.json"

    leaderboard = {}
    if path.exists():
        with open(path) as f:
            leaderboard = json.load(f)

    leaderboard.setdefault(model_id, {})[metric] = {
        "score": score,
        "total": total,
        "failed": failed,
    }

    with open(path, "w") as f:
        json.dump(leaderboard, f, indent=2)
