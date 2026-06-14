"""Portable AIPD post-2023 label extension."""

from .model import COMPONENTS, TwoBranchLSTM
from .score import score_embedding_arrays
from .text import first_long_chunk, normalize_text

__all__ = [
    "COMPONENTS",
    "TwoBranchLSTM",
    "first_long_chunk",
    "normalize_text",
    "score_embedding_arrays",
]
