import json
import random
import re
from typing import Literal, Optional, Type

from pydantic import BaseModel


try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# Output schemas — passed to generate() so vLLM can constrain token sampling.
class WinnerOutput(BaseModel):
    reasoning: str
    winner: Literal["A", "B", "tie"]


class ScoreOutput(BaseModel):
    reasoning: str
    score: float


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
        if self.backend == "fake":
            print(f"Loading {self.name} (fake backend — no model downloaded).")
            return

        if self.backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "vLLM is not installed. Install it or use --backend transformers."
                )
            from transformers import AutoTokenizer
            print(f"Loading {self.name} with vLLM...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
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

    def _apply_chat_template(self, prompt: str) -> str:
        """Format prompt as a chat message using the model's template.

        Passes enable_thinking=False for Qwen3 models to suppress <think> blocks
        and keep output short and parseable.
        """
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self._tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            # tokenizer doesn't support enable_thinking (non-Qwen3 model)
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _make_sampling_params(self, schema: Optional[Type[BaseModel]]):
        base = dict(temperature=0.0, max_tokens=5000)
        if schema is None:
            return SamplingParams(**base)
        try:
            from vllm.sampling_params import StructuredOutputsParams
            structured = StructuredOutputsParams(json=schema.model_json_schema())
            return SamplingParams(**base, structured_outputs=structured)
        except Exception:
            return SamplingParams(**base)

    def _generate_fake(self, prompt: str) -> str:
        """Return a plausible random JSON response without loading any model."""
        rng = random.Random(prompt)  # deterministic per prompt
        if '"winner"' in prompt:
            winner = rng.choice(["A", "B", "tie"])
            return json.dumps({"reasoning": "Fake reasoning.", "winner": winner})
        else:
            score = round(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]), 2)
            return json.dumps({"reasoning": "Fake reasoning.", "score": score})

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> str:
        if self.backend == "fake":
            return self._generate_fake(prompt)

        formatted = self._apply_chat_template(prompt)

        if self._llm is not None:
            params = self._make_sampling_params(schema)
            outputs = self._llm.generate([formatted], params)
            return outputs[0].outputs[0].text

        if self._model is not None:
            import torch
            inputs = self._tokenizer(formatted, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=5000,
                    do_sample=False,
                )
            return self._tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

        raise RuntimeError("Model not loaded. Call judge.load() first.")

    def parse_json(self, text: str) -> Optional[dict]:
        """Parse JSON from model output, stripping Qwen3 <think> blocks if present."""
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
