#!/usr/bin/env python3
"""Export and hash-check the frozen AIPD extension model bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "workspace/shared/aipd_classification/processed/"
    "aipd_ac_lstm_learning_curve_train05_fast1_20260515"
)
DEFAULT_MANIFEST = REPO_ROOT / "empirics/data/interim/aipd_ac_qwen_frozen_model_manifest_20260515.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "models").mkdir(exist_ok=True)

    for component, record in manifest["model_files"].items():
        source = args.source / "models" / f"{component}.pt"
        observed = sha256(source)
        if observed != record["sha256"]:
            raise ValueError(f"hash mismatch for {component}: {observed}")
        shutil.copy2(source, args.output / "models" / source.name)

    for name in ["thresholds.json", "frozen_model_manifest.json"]:
        source_name = "frozen_model_manifest.json" if name.startswith("frozen") else name
        shutil.copy2(args.source / source_name, args.output / name)
    print(f"Exported verified model bundle to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

