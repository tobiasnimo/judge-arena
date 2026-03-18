from .base import Judge


# Registry maps CLI judge IDs to (HuggingFace model_id, display_name)
REGISTRY: dict[str, tuple[str, str]] = {
    "fake":      ("fake/fake",          "Fake"),
    "qwen-0.8b": ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B"),
    "qwen-2b":   ("Qwen/Qwen3.5-2B",   "Qwen3.5-2B"),
    "qwen-4b":   ("Qwen/Qwen3.5-4B",   "Qwen3.5-4B"),
    "qwen-9b":   ("Qwen/Qwen3.5-9B",   "Qwen3.5-9B"),
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
