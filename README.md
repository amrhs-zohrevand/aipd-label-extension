# aipd-label-extension

The USPTO's [Artificial Intelligence Patent Dataset (AIPD)](https://www.uspto.gov/ip-policy/economic-research/research-datasets/artificial-intelligence-patent-dataset) classifies U.S. patent documents into eight AI component categories. Official labels cover documents published through 2023. This package extends that coverage to later documents using the same eight-component label structure, so studies built on AIPD can include more recent patent applications without switching classification systems.

The extension trains on official AIPD@0.93 labels and reproduces their logic: embed the abstract and claims separately with `Qwen/Qwen3-Embedding-8B`, score eight frozen two-branch LSTM models, and define `any_ai` as the OR over component predictions. Official labels are never overwritten — the extension only fills the post-2023 gap.

## Installation

Scoring and tests (no GPU required):

```bash
pip install -e ".[test]"
```

Full pipeline including the embedding step (requires a CUDA GPU):

```bash
pip install -e ".[gpu]"
```

## Getting the model weights

Download `model-bundle-v0.1.0.tar.gz` from the [Releases page](https://github.com/amrhs-zohrevand/aipd-label-extension/releases) and extract it into `models/release/`:

```bash
tar -xf model-bundle-v0.1.0.tar.gz -C models/release/
```

The bundle contains eight frozen `.pt` weight files and a `thresholds.json` with the component-specific decision thresholds selected during training.

## Usage

Input files must be CSV or Parquet with three columns: `doc_id`, `abstract`, and `claims`. The `claims` field should contain all claims joined in their original order with blank lines between them.

```bash
aipd-label-extension \
  --input path/to/documents.csv \
  --output path/to/labels.parquet \
  --model-bundle models/release \
  --backend vllm
```

See `examples/example_patents.csv` for a minimal input example.

### Output columns

Each output row contains:

| Column                                             | Description                                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| `aipd_qwen_ac_score_<component>`                   | Raw sigmoid score for each of the eight AIPD components                  |
| `aipd_qwen_ac_predict_<component>`                 | Binary prediction at the frozen component threshold                      |
| `aipd_qwen_ac_predict_any_ai`                      | OR over the eight component predictions (main label)                     |
| `aipd_qwen_ac_score_any_ai`                        | Maximum component score (diagnostic only, not a calibrated probability)  |
| `model_version`, `label_source`                    | Provenance metadata                                                      |
| `abstract_text_truncated`, `claims_text_truncated` | Whether the field was truncated before embedding                         |

## Performance

The model was selected at 5% of available training data (172,606 documents) after evaluating a learning curve on a locked 99,791-document holdout. Performance against official AIPD@0.93 labels:

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 92.57% |
| F1                | 0.787  |
| Average precision | 0.866  |
| AUROC             | 0.962  |

The `0.93` in AIPD@0.93 is the USPTO's score threshold used to produce the teacher labels — it is not this model's accuracy.

Component performance varies. Machine learning, NLP, speech, and vision components all reach F1 > 0.72. Evolutionary computation (F1 = 0.31) and knowledge processing (F1 = 0.58) are harder to reproduce, reflecting their rarity in the training data. For most research applications the `any_ai` OR label is the appropriate output; component-level predictions should be treated as diagnostics.

## Hardware

Embedding with `Qwen/Qwen3-Embedding-8B` requires a CUDA GPU with enough memory for an 8B-parameter model. The scoring step runs on CPU. A university cluster or cloud GPU instance works fine.

## Tests

```bash
pytest tests/ -v
```

The suite covers text normalization, chunking, the OR rule, CSV/Parquet I/O, missing-column detection, empty-field filtering, and a full read→filter→score→write pipeline.

## Citation

If you use this package, please also cite the underlying AIPD dataset:

> Giczy, A. V., Pairolero, N. A., & Toole, A. A. (2022). Identifying artificial intelligence (AI) invention: A novel AI patent dataset. *Journal of Technology Transfer*, 47(2), 476–505.
>
> Pairolero, N. A., et al. (2025). *Artificial Intelligence Patent Dataset: 2023 Update*. USPTO Office of the Chief Economist.
