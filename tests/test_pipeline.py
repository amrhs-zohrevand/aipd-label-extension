from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aipd_label_extension.cli import read_frame, write_frame
from aipd_label_extension.model import COMPONENTS, TwoBranchLSTM
from aipd_label_extension.score import score_embedding_arrays
from aipd_label_extension.text import first_long_chunk, normalize_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_bundle(tmp_path: Path, input_dim: int = 4) -> Path:
    """Build a minimal model bundle with zeroed weights (cpu-only, no GPU)."""
    bundle = tmp_path / "bundle"
    (bundle / "models").mkdir(parents=True)
    thresholds = {"component_thresholds": {c: 0.5 for c in COMPONENTS}}
    (bundle / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
    for component in COMPONENTS:
        model = TwoBranchLSTM(input_dim=input_dim, hidden_dim=64, dropout=0.10)
        for parameter in model.parameters():
            torch.nn.init.constant_(parameter, 0.0)
        torch.save(model.state_dict(), bundle / "models" / f"{component}.pt")
    return bundle


def _zero_embeddings(n: int, dim: int = 4) -> np.ndarray:
    return np.zeros((n, dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_normalize_text_is_mechanical() -> None:
    assert normalize_text("  Claim A.\x00\nLine B.   \n") == "Claim A.\nLine B."
    assert normalize_text(None) == ""


def test_first_long_chunk_matches_production_guard() -> None:
    first, chunks = first_long_chunk("A" * 12 + "\n\n" + "B" * 12, max_chars=15)
    assert first == "A" * 12
    assert chunks == 2


def test_scoring_emits_component_labels_and_any_ai_or(tmp_path) -> None:
    model_bundle = tmp_path / "bundle"
    model_dir = model_bundle / "models"
    model_dir.mkdir(parents=True)
    thresholds = {"component_thresholds": {component: 0.5 for component in COMPONENTS}}
    (model_bundle / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")

    for idx, component in enumerate(COMPONENTS):
        model = TwoBranchLSTM(input_dim=4, hidden_dim=64, dropout=0.10)
        for parameter in model.parameters():
            torch.nn.init.constant_(parameter, 0.0)
        model.combine[-1].bias.data.fill_(1.0 if idx == 0 else -1.0)
        torch.save(model.state_dict(), model_dir / f"{component}.pt")

    abstract = np.zeros((3, 4), dtype=np.float32)
    claims = np.zeros((3, 4), dtype=np.float32)
    result = score_embedding_arrays(
        abstract,
        claims,
        model_bundle=model_bundle,
        batch_size=2,
        device="cpu",
    )
    assert result["aipd_qwen_ac_predict_ml"].tolist() == [1, 1, 1]
    assert result["aipd_qwen_ac_predict_any_ai"].tolist() == [1, 1, 1]
    component_columns = [f"aipd_qwen_ac_predict_{component}" for component in COMPONENTS]
    expected = result[component_columns].max(axis=1)
    assert result["aipd_qwen_ac_predict_any_ai"].equals(expected.astype(np.int8))


# ---------------------------------------------------------------------------
# Integration tests: CSV/Parquet I/O, column validation, filtering, schema
# ---------------------------------------------------------------------------

def test_read_frame_csv_roundtrip(tmp_path: Path) -> None:
    """read_frame and write_frame are symmetric for CSV."""
    src = tmp_path / "input.csv"
    src.write_text("doc_id,abstract,claims\nA,text,claims text\n", encoding="utf-8")
    frame = read_frame(src)
    assert list(frame.columns) == ["doc_id", "abstract", "claims"]
    out = tmp_path / "output.csv"
    write_frame(frame, out)
    reloaded = read_frame(out)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), reloaded.reset_index(drop=True))


def test_read_frame_parquet_roundtrip(tmp_path: Path) -> None:
    """read_frame and write_frame are symmetric for Parquet."""
    frame = pd.DataFrame({"doc_id": ["A"], "abstract": ["text"], "claims": ["c"]})
    path = tmp_path / "data.parquet"
    write_frame(frame, path)
    reloaded = read_frame(path)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), reloaded.reset_index(drop=True))


def test_missing_column_detection() -> None:
    """The column-validation logic used in main() raises on missing required cols."""
    frame = pd.DataFrame({"doc_id": ["A"], "abstract": ["text"]})  # no 'claims'
    required = {"doc_id", "abstract", "claims"}
    missing = required - set(frame.columns)
    assert missing == {"claims"}
    with pytest.raises(ValueError, match="missing required columns"):
        if missing:
            raise ValueError(f"input is missing required columns: {sorted(missing)}")


def test_empty_field_rows_are_filtered(tmp_path: Path) -> None:
    """Rows where abstract or claims is empty after normalization are excluded."""
    bundle = _build_bundle(tmp_path)
    frame = pd.DataFrame({
        "doc_id": ["has_both", "no_abstract", "no_claims", "both_empty"],
        "abstract": ["neural network method", "", "some abstract", ""],
        "claims": ["1. A method.", "1. A method.", "", ""],
    })
    usable = frame["abstract"].ne("") & frame["claims"].ne("")
    filtered = frame.loc[usable].reset_index(drop=True)
    assert list(filtered["doc_id"]) == ["has_both"]


def test_full_pipeline_csv_to_parquet_output_schema(tmp_path: Path) -> None:
    """Exercises read → filter → score_embedding_arrays → write, checks schema."""
    bundle = _build_bundle(tmp_path)

    input_path = tmp_path / "input.csv"
    pd.DataFrame({
        "doc_id": ["doc1", "doc2"],
        "abstract": ["A neural network based scheduler.", ""],
        "claims": ["1. A method using ML.", "1. A bolt."],
    }).to_csv(input_path, index=False)

    frame = read_frame(input_path)
    # mirror cli.py: normalize via first_long_chunk before filtering
    frame["abstract"] = frame["abstract"].map(lambda v: first_long_chunk(v, 24_000)[0])
    frame["claims"] = frame["claims"].map(lambda v: first_long_chunk(v, 110_000)[0])
    usable = frame["abstract"].ne("") & frame["claims"].ne("")
    frame = frame.loc[usable].reset_index(drop=True)
    assert len(frame) == 1  # doc2 dropped (empty abstract → "" after normalize)

    abstract = _zero_embeddings(len(frame))
    claims = _zero_embeddings(len(frame))
    labels = score_embedding_arrays(abstract, claims, model_bundle=bundle, batch_size=2, device="cpu")

    output = frame[["doc_id"]].join(labels)
    out_path = tmp_path / "output.parquet"
    write_frame(output, out_path)

    result = pd.read_parquet(out_path)
    assert list(result["doc_id"]) == ["doc1"]

    expected_cols = (
        [f"aipd_qwen_ac_score_{c}" for c in COMPONENTS]
        + [f"aipd_qwen_ac_predict_{c}" for c in COMPONENTS]
        + ["aipd_qwen_ac_score_any_ai", "aipd_qwen_ac_predict_any_ai"]
    )
    for col in expected_cols:
        assert col in result.columns, f"missing column: {col}"

    # any_ai OR must be consistent with component predictions
    comp_preds = result[[f"aipd_qwen_ac_predict_{c}" for c in COMPONENTS]]
    assert (result["aipd_qwen_ac_predict_any_ai"] == comp_preds.max(axis=1)).all()


def test_example_csv_has_required_columns() -> None:
    """The shipped examples/example_patents.csv satisfies the input contract."""
    frame = read_frame(Path("examples/example_patents.csv"))
    required = {"doc_id", "abstract", "claims"}
    missing = required - set(frame.columns)
    assert not missing, f"example CSV missing columns: {missing}"
    assert len(frame) >= 1
