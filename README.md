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
| `datasets/context.json` | [FeedbackQA](https://github.com/McGill-NLP/feedbackqa) |

## Supported Models

| CLI ID | Model |
|---|---|
| `qwen-0.8b` | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| `qwen-2b` | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| `qwen-4b` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| `qwen-9b` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |

Adding a new model only requires adding one entry to `src/inference/registry.py`.

## Setup

```bash
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
```

Results are written to `leaderboard/` after each run.

## Leaderboard

Each metric produces its own leaderboard file. An overall leaderboard is computed when all three metrics have been run for a model.

```
leaderboard/
├── bestof.json       # sorted by accuracy (desc)
├── conversation.json # sorted by MAE (asc)
├── context.json      # sorted by MAE (asc)
└── overall.json      # sorted by overall_score (desc)
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
├── leaderboard/       # Auto-generated leaderboard JSONs (created on first run)
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
