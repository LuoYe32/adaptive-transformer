import torch
import torch.nn as nn
from torch import Tensor

from src.data.quality_metrics import (
    fraction_missing,
    snr_mad,
    local_variance,
    kurtosis,
    fraction_outliers,
    global_quality_summary,
)


def compute_quality_features(x: Tensor, mask: Tensor) -> Tensor:
    """
    Compute deterministic quality features from input data.

    Args:
        x: Tensor[B, C, T]      -- input signal
        mask: Tensor[B, C, T]   -- missingness mask (1 = observed, 0 = missing)

    Returns:
        Tensor[B, F] where F is the total number of quality features
    """
    fm = fraction_missing(mask)
    snr = snr_mad(x, mask)
    var = local_variance(x, mask)
    kurt = kurtosis(x, mask)
    out = fraction_outliers(x, mask)

    per_channel = torch.stack([fm, snr, var, kurt, out], dim=-1)

    global_feats = global_quality_summary(per_channel)

    B, C, F_c = per_channel.shape
    per_channel_flat = per_channel.view(B, C * F_c)

    features = torch.cat([per_channel_flat, global_feats], dim=-1)

    return features


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder for quality features.
    """

    def __init__(self, d_in: int, d_q: int, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_q),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DataQualityModule(nn.Module):
    """
    Data Quality Module (DQM).

    Converts raw data and missingness masks into a learned quality embedding q.

    Pipeline:
        (x, mask)
            -> deterministic quality features
            -> MLP encoder
            -> q ∈ R^{d_q}
    """

    def __init__(self, d_q: int, n_channels: int, n_metrics: int = 5):
        """
        Args:
            d_q: dimension of the output quality embedding
            n_channels: number of channels in the input data
            n_metrics: number of per-channel quality metrics
        """
        super().__init__()

        d_in = n_channels * n_metrics + 3 * n_metrics

        self.d_in = d_in
        self.d_q = d_q

        self.encoder = MLPEncoder(d_in=d_in, d_q=d_q)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Tensor[B, C, T]
            mask: Tensor[B, C, T]

        Returns:
            q: Tensor[B, d_q]
        """
        feats = compute_quality_features(x, mask)
        q = self.encoder(feats)
        return q
