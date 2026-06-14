"""Score paired abstract and claims embeddings with the frozen models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import COMPONENTS, load_component_models


def load_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("component_thresholds", payload)
    return {component: float(values[component]) for component in COMPONENTS}


def score_embedding_arrays(
    abstract_embeddings: np.ndarray,
    claims_embeddings: np.ndarray,
    *,
    model_bundle: Path,
    batch_size: int = 4096,
    device: str | None = None,
) -> pd.DataFrame:
    """Return component scores/predictions and their any-AI OR."""

    if abstract_embeddings.shape != claims_embeddings.shape:
        raise ValueError("abstract and claims embeddings must have identical shapes")
    if abstract_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [documents, dimensions]")

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    thresholds = load_thresholds(model_bundle / "thresholds.json")
    models = load_component_models(
        model_bundle / "models",
        device=torch_device,
        input_dim=abstract_embeddings.shape[1],
    )

    abstract = torch.from_numpy(np.asarray(abstract_embeddings, dtype=np.float32)).reshape(
        len(abstract_embeddings), 1, abstract_embeddings.shape[1]
    )
    claims = torch.from_numpy(np.asarray(claims_embeddings, dtype=np.float32)).reshape(
        len(claims_embeddings), 1, claims_embeddings.shape[1]
    )
    lengths = torch.ones(len(abstract_embeddings), dtype=torch.int64)

    output: dict[str, np.ndarray] = {}
    component_predictions: list[np.ndarray] = []
    component_scores: list[np.ndarray] = []
    for component, model in models.items():
        scores = np.zeros(len(abstract_embeddings), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(scores), batch_size):
                stop = min(start + batch_size, len(scores))
                logits = model(
                    abstract[start:stop].to(torch_device),
                    lengths[start:stop].to(torch_device),
                    claims[start:stop].to(torch_device),
                    lengths[start:stop].to(torch_device),
                )
                scores[start:stop] = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        predictions = (scores >= thresholds[component]).astype(np.int8)
        output[f"aipd_qwen_ac_score_{component}"] = scores
        output[f"aipd_qwen_ac_predict_{component}"] = predictions
        component_scores.append(scores)
        component_predictions.append(predictions)

    output["aipd_qwen_ac_score_any_ai"] = np.max(np.column_stack(component_scores), axis=1)
    output["aipd_qwen_ac_predict_any_ai"] = np.max(
        np.column_stack(component_predictions), axis=1
    ).astype(np.int8)
    return pd.DataFrame(output)

