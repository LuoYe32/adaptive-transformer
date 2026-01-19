"""
Basic shape tests for Phase 1 components.

These tests ensure that:
- DataQualityModule accepts inputs of expected shape
- Output embedding q has the correct dimensionality
- The forward pass works without errors
"""

import torch

from src.models.dqm import DataQualityModule, compute_quality_features


def test_compute_quality_features_shape():
    """
    Test that deterministic quality feature extraction
    returns the expected feature shape.
    """
    B, C, T = 2, 16, 128

    x = torch.randn(B, C, T)
    mask = torch.ones_like(x)

    feats = compute_quality_features(x, mask)

    # 5 per-channel metrics + 3 global aggregates per metric
    n_metrics = 5
    expected_dim = C * n_metrics + 3 * n_metrics

    assert feats.shape == (B, expected_dim)


def test_dqm_output_shape():
    """
    Test that DataQualityModule produces a quality embedding
    of the expected shape.
    """
    B, C, T = 2, 16, 128
    d_q = 8

    x = torch.randn(B, C, T)
    mask = torch.ones_like(x)

    dqm = DataQualityModule(
        d_q=d_q,
        n_channels=C,
    )

    q = dqm(x, mask)

    assert q.shape == (B, d_q)


def test_dqm_forward_with_missing_values():
    """
    Test that DQM can handle missing values (mask != 1 everywhere).
    """
    B, C, T = 2, 16, 128
    d_q = 8

    x = torch.randn(B, C, T)
    mask = (torch.rand(B, C, T) > 0.3).float()

    dqm = DataQualityModule(
        d_q=d_q,
        n_channels=C,
    )

    q = dqm(x, mask)

    assert q.shape == (B, d_q)
    assert torch.isfinite(q).all()
