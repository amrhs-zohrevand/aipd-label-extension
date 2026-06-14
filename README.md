# AIPD post-2023 label extension

This folder is the release candidate for the code that extends the USPTO
Artificial Intelligence Patent Dataset (AIPD) beyond its official 2023
publication coverage. It is intentionally self-contained and does not assume
access to Snellius, the project repository, or any particular cloud provider.

The classifier reproduces the project's frozen extension procedure:

1. Preserve each published application's abstract and ordered claims as two
   separate text fields.
2. Apply only mechanical text hygiene: remove NUL bytes, normalize line
   endings, and trim outer whitespace.
3. Apply the production long-chunk guards (24,000 characters for abstracts and
   110,000 for claims), retaining the first chunk used by the frozen scorer.
4. Embed the abstract and claims separately with
   `Qwen/Qwen3-Embedding-8B`, using the model's pooled 4,096-dimensional
   output and a 32,768-token context window.
5. Score eight AIPD component models, each with separate abstract and claims
   LSTM branches followed by a neural combination layer.
6. Apply the frozen component-specific thresholds and define `any_ai` as the
   logical OR over the eight component predictions.

The model is an AIPD-compatible extension, not an official USPTO label. Inside
official AIPD coverage, the official USPTO label remains authoritative.

## What is included

- `src/aipd_label_extension/`: portable text-to-label pipeline
- `scripts/export_model_bundle.py`: exports the frozen weights, thresholds,
  and manifest from the research project into a release bundle
- `examples/example_patents.csv`: minimal input schema
- `tests/`: tests for text handling, model scoring, and the any-AI rule

The 66 MB frozen weight bundle is deliberately not stored in this Git
repository. Before public release it should be attached to a GitHub release or
placed in a versioned model repository. The export script creates the exact
bundle from the research project's frozen artifacts.

## Hardware

The reference embedding backend uses `vLLM` on a CUDA GPU. It is suitable for
a paid cloud GPU, a university cluster, or a Linux workstation with enough GPU
memory for Qwen3-Embedding-8B. The classifier itself is small; embedding the
text is the resource-intensive step. CPU-only execution is technically
possible with a compatible backend, but is not a practical route for a large
patent corpus.

## Installation

Core scoring and tests:

```bash
python -m pip install -e ".[test]"
```

Reference text-to-label pipeline on a CUDA/Linux machine:

```bash
python -m pip install -e ".[gpu]"
```

## Export the frozen model

From the main research repository:

```bash
python workspace/sandbox/aipd-label-extension-replication/scripts/export_model_bundle.py \
  --output workspace/sandbox/aipd-label-extension-replication/models/release
```

The export script verifies every weight file against the hashes in the frozen
model manifest before it writes the bundle.

## Label a file

Inputs may be CSV or Parquet and must contain `doc_id`, `abstract`, and
`claims`. The `claims` field should contain all claims in their original
sequence, joined with blank lines.

```bash
aipd-label-extension \
  --input examples/example_patents.csv \
  --output outputs/example_labels.parquet \
  --model-bundle models/release \
  --backend vllm
```

The output contains:

- one raw score and one binary prediction for each AIPD component;
- `aipd_qwen_ac_score_any_ai`, the maximum component score, retained only as a
  diagnostic because it is not a calibrated probability;
- `aipd_qwen_ac_predict_any_ai`, the OR over component predictions;
- model, text-version, and truncation metadata.

## Interpretation

The selected model was trained on 172,606 official-label documents and
evaluated on a locked, population-weighted 99,791-document holdout. Against
official AIPD@0.93 labels, it achieved 92.57% accuracy, 0.787 F1, 0.866 average
precision, and 0.962 AUROC. The `0.93` in AIPD@0.93 is the USPTO teacher-label
score threshold; it is not the accuracy of this extension model.

The public repository URL will be added to the paper appendix when this
release candidate is moved into its own GitHub repository and the frozen model
bundle is deposited.
