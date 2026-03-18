"""Shared leaderboard update logic used by all metrics and run_judge.py."""

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).parent.parent
LEADERBOARD_DIR = ROOT / "leaderboard"


def update_metric_leaderboard(metric: str, result: dict, sort_key: str, ascending: bool = False):
    """Upsert a result entry into a per-metric leaderboard JSON file."""
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    path = LEADERBOARD_DIR / f"{metric}.json"

    if path.exists():
        with open(path) as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {"metric": metric, "results": []}

    existing = next(
        (r for r in leaderboard["results"] if r["model_id"] == result["model_id"]), None
    )
    if existing:
        existing.update(result)
    else:
        leaderboard["results"].append(result)

    leaderboard["results"].sort(
        key=lambda r: r.get(sort_key) or 0, reverse=not ascending
    )
    leaderboard["last_updated"] = _now()

    with open(path, "w") as f:
        json.dump(leaderboard, f, indent=2)


def update_overall_leaderboard(model_id: str, model_name: str, updates: dict):
    """Merge new metric results into the overall leaderboard for a given model.

    `updates` should be a subset of keys:
        bestof_accuracy, conversation_mae, context_mae
    """
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    path = LEADERBOARD_DIR / "overall.json"

    if path.exists():
        with open(path) as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {"metric": "overall", "results": []}

    existing = next(
        (r for r in leaderboard["results"] if r["model_id"] == model_id), None
    )
    if existing is None:
        existing = {"model_name": model_name, "model_id": model_id}
        leaderboard["results"].append(existing)

    existing.update(updates)
    existing["last_run"] = _now()

    # Compute overall score from whichever metrics are available.
    # All components are normalised to [0, 1] where higher = better.
    components = []
    if existing.get("bestof_accuracy") is not None:
        components.append(existing["bestof_accuracy"])
    if existing.get("conversation_mae") is not None:
        components.append(1 - existing["conversation_mae"])
    if existing.get("context_mae") is not None:
        components.append(1 - existing["context_mae"])

    existing["overall_score"] = round(sum(components) / len(components), 4) if components else None

    leaderboard["results"].sort(
        key=lambda r: r.get("overall_score") or 0, reverse=True
    )
    leaderboard["last_updated"] = _now()

    with open(path, "w") as f:
        json.dump(leaderboard, f, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
