"""Text handling that matches the frozen research pipeline."""

from __future__ import annotations

import math


def normalize_text(value: object) -> str:
    """Apply mechanical hygiene without changing technical language."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).replace("\x00", " ")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def first_long_chunk(value: object, max_chars: int) -> tuple[str, int]:
    """Return the first production-style long chunk and total chunk count."""

    text = normalize_text(value)
    if not text:
        return "", 0
    if len(text) <= max_chars:
        return text, 1
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if split_at > start + int(max_chars * 0.60):
                end = split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks[0], len(chunks)
