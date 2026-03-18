from .base import Judge


# Registry maps CLI judge IDs to (HuggingFace model_id, display_name)
REGISTRY: dict[str, tuple[str, str]] = {
    "fake":      ("fake/fake",          "Fake"),
    "qwen-0.8b": ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B"),
    "qwen-2b":   ("Qwen/Qwen3.5-2B",   "Qwen3.5-2B"),
    "qwen-4b":   ("Qwen/Qwen3.5-4B",   "Qwen3.5-4B"),
    "qwen-9b":       ("Qwen/Qwen3.5-9B",                    "Qwen3.5-9B"),
    "nemotron-4b":   ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "Nemotron-3-Nano-4B"),
    "gemma-3-4b":    ("google/gemma-3-4b-it",                 "Gemma-3-4B"),
    "phi-4-mini":          ("microsoft/Phi-4-mini-instruct",        "Phi-4-Mini"),
    "phi-4-mini-flash":    ("microsoft/Phi-4-mini-flash-reasoning", "Phi-4-Mini-Flash"),
    "lfm2.5-1.2b":         ("LiquidAI/LFM2.5-1.2B-Instruct",       "LFM2.5-1.2B"),
    "lfm2.5-1.2b-think":   ("LiquidAI/LFM2.5-1.2B-Thinking",       "LFM2.5-1.2B-Thinking"),
}


def load_judge(judge_id: str, backend: str = "vllm") -> Judge:
    if judge_id not in REGISTRY:
        raise ValueError(
            f"Unknown judge '{judge_id}'. Available: {list(REGISTRY.keys())}"
        )
    model_id, name = REGISTRY[judge_id]
    # fake judge always uses the fake backend regardless of --backend flag
    resolved_backend = "fake" if judge_id == "fake" else backend
    judge = Judge(model_id=model_id, name=name, backend=resolved_backend)
    judge.load()
    return judge


def list_judges() -> list[str]:
    return list(REGISTRY.keys())
