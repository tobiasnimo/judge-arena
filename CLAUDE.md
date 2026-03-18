# Judge Arena — Agent Guide

## What this repo does

Benchmarks LLMs as judges. A judge model is given structured evaluation tasks and its outputs are scored against ground truth. Results feed into a persistent leaderboard at `results/leaderboard.json`.

## Running locally (no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_judge.py --judge fake --metric all
```

The `fake` judge returns deterministic random JSON without downloading any model. Always use it to verify code changes before running real models.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--judge` | required | Judge ID from registry (e.g. `fake`, `qwen-2b`) |
| `--metric` | `all` | `bestof`, `conversation`, `context`, or `all` |
| `--backend` | `vllm` | `vllm` or `transformers` |
| `--debug` | off | Adds `raw` field (raw LLM output) to every row in the results JSON |

## Repository layout

```
src/
├── run_judge.py          # CLI entry point — start here
├── leaderboard.py        # update_leaderboard() and save_metric_results()
├── inference/
│   ├── base.py           # Judge class: load(), generate(), parse_json()
│   │                     # Also defines WinnerOutput and ScoreOutput Pydantic schemas
│   └── registry.py       # REGISTRY dict: CLI id → (hf_model_id, display_name)
└── metrics/
    ├── bestof.py         # Accuracy: pick the better of two responses
    ├── conversation.py   # MAE: score a generated answer vs ground truth
    └── context.py        # MAE: score how helpful a context passage is
datasets/                 # 100-sample JSON datasets (do not modify)
results/                  # Git-ignored. Created on first run.
    ├── bestof/           # Per-model per-item results
    ├── conversation/
    ├── context/
    └── leaderboard.json  # Single leaderboard, keyed by model_id
```

## Key design decisions

**Inference (`src/inference/base.py`)**
- Chat template is always applied via `_apply_chat_template()` before calling the model. This is critical — raw prompts don't work with instruct models.
- `enable_thinking=False` is passed to suppress Qwen3 `<think>` blocks. Falls back gracefully for non-Qwen models.
- Structured outputs use `StructuredOutputsParams` (vLLM stable API, not the deprecated `GuidedDecodingParams`). This constrains token sampling to valid JSON matching the Pydantic schema.
- `generate(prompt, schema=WinnerOutput)` or `generate(prompt, schema=ScoreOutput)` — each metric passes its own schema.

**Metrics (`src/metrics/*.py`)**
- Each metric's `run(judge, debug=False)` loads its dataset, calls `judge.generate()`, calls `judge.parse_json()`, accumulates rows, then calls `save_metric_results()` and `update_leaderboard()`.
- Each row always has: `question`, `reasoning`, `predicted_*`, `actual_*`, `parseable`. With `--debug`: also `raw`.
- `parseable=False` rows count as `failed` in the leaderboard but are still written to the results file.

**Leaderboard (`src/leaderboard.py`)**
- Single file: `results/leaderboard.json`
- Keyed by `model_id` (HuggingFace ID, e.g. `"Qwen/Qwen3.5-2B"`)
- Metric field names: `accuracy` (bestof), `mae` (conversation, context)
- `overall.score` is recomputed after every metric run: `mean([bestof_accuracy, 1 - conversation_mae, 1 - context_mae])`
- File is re-sorted by `overall.score` descending on every write

**Adding a new model**
Add one entry to `REGISTRY` in `src/inference/registry.py`:
```python
"my-model": ("org/model-id-on-hf", "Display Name"),
```

**Adding a new metric**
1. Create `src/metrics/mymetric.py` with a `run(judge, debug=False) -> dict` function following the pattern of the existing metrics
2. Import and register it in `src/run_judge.py`
3. Add the metric field name mapping to `_METRIC_FIELD` in `src/leaderboard.py`

## Leaderboard structure

```json
{
  "Qwen/Qwen3.5-2B": {
    "bestof":       { "accuracy": 0.61, "total": 100, "failed": 2 },
    "conversation": { "mae": 0.18,      "total": 100, "failed": 1 },
    "context":      { "mae": 0.21,      "total": 100, "failed": 0 },
    "overall":      { "score": 0.71 }
  }
}
```

## Known issues / active work

- Unparseable rate with real models is being investigated. Structured outputs (`StructuredOutputsParams`) should eliminate this but hasn't been confirmed working end-to-end on real hardware yet.
- Use `--debug` to inspect raw model output when diagnosing parse failures.
