"""Shared leaderboard and results persistence used by all metrics."""

import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

# Maps metric name → the field name used in the leaderboard entry
_METRIC_FIELD = {
    "bestof":       "accuracy",
    "conversation": "mae",
    "context":      "mae",
}


def save_metric_results(metric: str, model_id: str, rows: list):
    """Write per-item judge responses to results/[metric]/[model_id].json."""
    out_dir = RESULTS_DIR / metric
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_id.replace('/', '--')}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def update_leaderboard(model_id: str, metric: str, score, total: int, failed: int):
    """Upsert a metric result for a model and recompute overall score.

    Leaderboard structure:
    {
      "<model_id>": {
        "bestof":       {"accuracy": ..., "total": ..., "failed": ...},
        "conversation": {"mae": ...,      "total": ..., "failed": ...},
        "context":      {"mae": ...,      "total": ..., "failed": ...},
        "overall":      {"score": ...}
      },
      ...         <- sorted by overall score descending
    }
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "leaderboard.json"

    leaderboard = {}
    if path.exists():
        with open(path) as f:
            leaderboard = json.load(f)

    field = _METRIC_FIELD[metric]
    leaderboard.setdefault(model_id, {})[metric] = {
        field:   score,
        "total": total,
        "failed": failed,
    }

    _recompute_overall(leaderboard[model_id])

    # Sort models by overall score descending
    leaderboard = dict(
        sorted(
            leaderboard.items(),
            key=lambda kv: (kv[1].get("overall") or {}).get("score") or 0,
            reverse=True,
        )
    )

    with open(path, "w") as f:
        json.dump(leaderboard, f, indent=2)


def _recompute_overall(entry: dict):
    """Compute overall_score from whichever metrics are present and store it."""
    components = []
    if "bestof" in entry:
        components.append(entry["bestof"]["accuracy"])
    if "conversation" in entry:
        components.append(1 - entry["conversation"]["mae"])
    if "context" in entry:
        components.append(1 - entry["context"]["mae"])

    entry["overall"] = {
        "score": round(sum(components) / len(components), 4) if components else None
    }
