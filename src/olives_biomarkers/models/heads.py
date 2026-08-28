"""Classification heads."""

from __future__ import annotations

import torch
from torch import nn


class MultiLabelHead(nn.Module):
    """Dropout + linear head returning **raw logits**.

    Logits, never probabilities: ``BCEWithLogitsLoss`` needs them, and the
    dropout placed here is what Monte Carlo dropout reactivates at inference.

    Args:
        input_dim: Width of the incoming embedding.
        n_labels: Number of biomarkers to predict.
        hidden_dim: Optional hidden layer width; None for a single linear layer.
        dropout: Dropout probability, active during MC dropout inference.
    """

    def __init__(
        self,
        input_dim: int,
        n_labels: int,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_labels = n_labels

        if hidden_dim:
            self.network = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_labels),
            )
        else:
            self.network = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, n_labels))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, n_labels)`` raw logits."""
        return self.network(embedding)
