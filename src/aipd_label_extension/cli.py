"""Command-line interface for text-to-label scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .embed import VLLMQwenEmbedder
from .score import score_embedding_arrays
from .text import first_long_chunk


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False, compression="zstd")
    else:
        frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-bundle", required=True, type=Path)
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--abstract-column", default="abstract")
    parser.add_argument("--claims-column", default="claims")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--scoring-batch-size", type=int, default=4096)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = read_frame(args.input)
    required = {args.id_column, args.abstract_column, args.claims_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    abstract_chunks = frame[args.abstract_column].map(lambda value: first_long_chunk(value, 24_000))
    claims_chunks = frame[args.claims_column].map(lambda value: first_long_chunk(value, 110_000))
    frame[args.abstract_column] = abstract_chunks.map(lambda value: value[0])
    frame[args.claims_column] = claims_chunks.map(lambda value: value[0])
    frame["abstract_chunk_count"] = abstract_chunks.map(lambda value: value[1])
    frame["claims_chunk_count"] = claims_chunks.map(lambda value: value[1])
    usable = frame[args.abstract_column].ne("") & frame[args.claims_column].ne("")
    if not usable.all():
        frame = frame.loc[usable].copy()

    embedder = VLLMQwenEmbedder()
    abstract = embedder.encode(frame[args.abstract_column].tolist(), batch_size=args.embedding_batch_size)
    claims = embedder.encode(frame[args.claims_column].tolist(), batch_size=args.embedding_batch_size)
    labels = score_embedding_arrays(
        abstract.vectors,
        claims.vectors,
        model_bundle=args.model_bundle,
        batch_size=args.scoring_batch_size,
        device=args.device,
    )

    manifest_path = args.model_bundle / "frozen_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    output = frame[[args.id_column]].reset_index(drop=True).join(labels)
    output["abstract_text_truncated"] = abstract.truncated
    output["claims_text_truncated"] = claims.truncated
    output["abstract_chunk_count"] = frame["abstract_chunk_count"].to_numpy()
    output["claims_chunk_count"] = frame["claims_chunk_count"].to_numpy()
    output["abstract_input_tokens_original"] = abstract.original_tokens
    output["claims_input_tokens_original"] = claims.original_tokens
    output["model_version"] = manifest.get("model_version", "unknown")
    output["label_source"] = "project_qwen_ac_extension"
    write_frame(output, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
