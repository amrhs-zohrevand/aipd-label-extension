"""Reference Qwen embedding backend used by the frozen extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EmbeddedText:
    vectors: np.ndarray
    truncated: np.ndarray
    original_tokens: np.ndarray
    used_tokens: np.ndarray


class VLLMQwenEmbedder:
    """Embed text with vLLM's Qwen pooling path, matching production."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Embedding-8B",
        *,
        max_model_len: int = 32768,
        token_buffer: int = 128,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "auto",
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.max_input_tokens = max_model_len - token_buffer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = LLM(
            model=model_id,
            runner="pooling",
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

    def _truncate(self, text: str) -> tuple[str, int, int, bool]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        original = len(token_ids)
        if original <= self.max_input_tokens:
            return text, original, original, False
        used = token_ids[: self.max_input_tokens]
        return self.tokenizer.decode(used, skip_special_tokens=True), original, len(used), True

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> EmbeddedText:
        prepared: list[str] = []
        original_tokens: list[int] = []
        used_tokens: list[int] = []
        truncated: list[bool] = []
        for text in texts:
            used_text, original, used, was_truncated = self._truncate(text)
            prepared.append(used_text)
            original_tokens.append(original)
            used_tokens.append(used)
            truncated.append(was_truncated)

        vectors: list[list[float]] = []
        for start in range(0, len(prepared), batch_size):
            outputs = self.model.embed(prepared[start : start + batch_size])
            vectors.extend(output.outputs.embedding for output in outputs)
        return EmbeddedText(
            vectors=np.asarray(vectors, dtype=np.float32),
            truncated=np.asarray(truncated, dtype=bool),
            original_tokens=np.asarray(original_tokens, dtype=np.int32),
            used_tokens=np.asarray(used_tokens, dtype=np.int32),
        )

