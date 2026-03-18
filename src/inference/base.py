import json
import re
from typing import Optional


try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class Judge:
    """LLM judge that loads a model once and exposes a generate() interface."""

    def __init__(self, model_id: str, name: str, backend: str = "vllm"):
        self.model_id = model_id
        self.name = name
        self.backend = backend
        self._llm = None        # vLLM handle
        self._model = None      # transformers model
        self._tokenizer = None  # transformers tokenizer

    def load(self):
        if self.backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "vLLM is not installed. Install it or use --backend transformers."
                )
            print(f"Loading {self.name} with vLLM...")
            self._llm = LLM(
                model=self.model_id,
                dtype="auto",
                trust_remote_code=True,
            )
        else:
            print(f"Loading {self.name} with transformers...")
            self._load_transformers()
        print("Model ready.")

    def _load_transformers(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def generate(self, prompt: str) -> str:
        if self._llm is not None:
            params = SamplingParams(temperature=0.0, max_tokens=512)
            outputs = self._llm.generate([prompt], params)
            return outputs[0].outputs[0].text

        if self._model is not None:
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            return self._tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

        raise RuntimeError("Model not loaded. Call judge.load() first.")

    def parse_json(self, text: str) -> Optional[dict]:
        """Parse JSON from model output, stripping Qwen3 <think> blocks if present."""
        # Strip Qwen3 thinking tokens before parsing
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract first JSON object if the model added surrounding text
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None
