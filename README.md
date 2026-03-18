# Judge Arena

A benchmarking tool to evaluate LLMs as judges. Models are scored on three distinct evaluation tasks using curated 100-sample datasets, and results are tracked in a persistent leaderboard.

## Tasks

| Metric | Task | Score |
|---|---|---|
| **Best-of-Two** | Given a question and two responses (A/B), pick the better one | Accuracy |
| **Conversation** | Given a question, a generated answer, and a ground truth, rate answer quality (0–1) | MAE |
| **Context** | Given a question and a context passage, rate how helpful the context is (0–1) | MAE |

For MAE-based tasks, lower is better. For accuracy, higher is better. The overall leaderboard normalises all metrics to [0, 1] (higher = better) and averages them.

## Datasets

Pre-loaded 100-sample datasets in `datasets/`. Loaders (Jupyter notebooks) are in `loaders/` if you need to regenerate them.

| File | Source |
|---|---|
| `datasets/bestof.json` | [LMSYS MT-Bench Human Judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) |
| `datasets/conversational.json` | [NVIDIA Judges Verdict](https://huggingface.co/datasets/nvidia/judges-verdict) |
| `datasets/context.json` | [FeedbackQA](https://huggingface.co/datasets/McGill-NLP/feedbackQA) |

## Supported Models

| CLI ID | Model | VRAM (bf16) |
|---|---|---|
| `fake` | Built-in fake judge (no model download, for local testing) | — |
| `qwen-0.8b` | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | ~2 GB |
| `qwen-2b` | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | ~5 GB |
| `qwen-4b` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | ~9 GB |
| `qwen-9b` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | ~19 GB |
| `nemotron-4b` | [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) | ~9 GB |
| `gemma-3-4b` | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | ~9 GB |
| `phi-4-mini` | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | ~8 GB |
| `phi-4-mini-flash` ⚙️ | [microsoft/Phi-4-mini-flash-reasoning](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning) | ~8 GB |
| `lfm2.5-1.2b` | [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | ~3 GB |
| `lfm2.5-1.2b-think` ⚙️ | [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) | ~3 GB |

> **⚙️ Reasoning models** emit chain-of-thought tokens before their JSON answer. The parser strips known thinking-block formats and falls back to extracting the first `{...}` object, but unparseable rates may be higher than with standard instruct models. Use `--debug` to inspect raw output if needed.

Adding a new model only requires adding one entry to `src/inference/registry.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

vLLM is the default inference backend. If vLLM is unavailable, fall back to `transformers` via `--backend transformers`.

## Usage

```bash
# Run all metrics for a judge
python src/run_judge.py --judge qwen-2b --metric all

# Run a single metric
python src/run_judge.py --judge qwen-9b --metric bestof

# Use transformers backend instead of vLLM
python src/run_judge.py --judge qwen-4b --metric conversation --backend transformers

# Test locally without downloading any model
python src/run_judge.py --judge fake --metric all

# Debug mode: include raw LLM output in results JSON
python src/run_judge.py --judge qwen-2b --metric bestof --debug
```

Results are written to `results/` after each run.

## Results & Leaderboard

Per-run judge responses (with reasoning) are saved alongside a single overall leaderboard.

```
results/
├── bestof/
│   └── Qwen--Qwen3.5-2B.json   # per-item: question, reasoning, predicted/actual winner, correct, parseable
├── conversation/
│   └── Qwen--Qwen3.5-2B.json   # per-item: question, reasoning, predicted/actual score, error, parseable
├── context/
│   └── Qwen--Qwen3.5-2B.json   # per-item: question, reasoning, predicted/actual score, error, parseable
└── leaderboard.json             # overall leaderboard keyed by model_id, sorted by overall score
```

With `--debug`, each item also includes a `raw` field with the unprocessed model output.

Leaderboard structure:
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

Overall score formula:
```
overall_score = mean([bestof_accuracy, 1 - conversation_mae, 1 - context_mae])
```

## Project Structure

```
judge-arena/
├── datasets/          # Pre-loaded evaluation datasets (100 samples each)
├── loaders/           # Jupyter notebooks used to generate the datasets
├── results/           # Auto-generated on first run: per-item responses + leaderboard
├── src/
│   ├── run_judge.py   # CLI entry point
│   ├── leaderboard.py # Leaderboard update logic
│   ├── inference/
│   │   ├── base.py    # Judge class (vLLM + transformers backends)
│   │   └── registry.py# Model registry and loader
│   └── metrics/
│       ├── bestof.py
│       ├── conversation.py
│       └── context.py
└── requirements.txt
```
