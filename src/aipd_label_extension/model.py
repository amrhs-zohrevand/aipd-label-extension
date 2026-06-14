"""Frozen two-branch classifier architecture."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


COMPONENTS = ("ml", "evo", "nlp", "speech", "vision", "planning", "kr", "hardware")


class TwoBranchLSTM(nn.Module):
    """Encode abstract and claims separately, then combine their representations."""

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 64, dropout: float = 0.10) -> None:
        super().__init__()
        self.abstract_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.claims_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def encode(lstm: nn.LSTM, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            values,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = lstm(packed)
        return hidden[-1]

    def forward(
        self,
        abstract: torch.Tensor,
        abstract_lengths: torch.Tensor,
        claims: torch.Tensor,
        claims_lengths: torch.Tensor,
    ) -> torch.Tensor:
        abstract_hidden = self.encode(self.abstract_lstm, abstract, abstract_lengths)
        claims_hidden = self.encode(self.claims_lstm, claims, claims_lengths)
        return self.combine(torch.cat([abstract_hidden, claims_hidden], dim=1)).squeeze(1)


def load_component_models(
    model_dir: Path,
    *,
    device: torch.device,
    input_dim: int = 4096,
    hidden_dim: int = 64,
    dropout: float = 0.10,
) -> Mapping[str, TwoBranchLSTM]:
    """Load the eight frozen component models."""

    models: dict[str, TwoBranchLSTM] = {}
    for component in COMPONENTS:
        model = TwoBranchLSTM(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
        state = torch.load(model_dir / f"{component}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models[component] = model
    return models

