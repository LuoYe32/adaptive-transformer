from typing import Optional

import torch.nn.functional as F
from torch import Tensor


def mse_loss(
    pred: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Mean Squared Error loss.

    Args:
        pred: Tensor[B, ...]
        target: Tensor[B, ...]
        reduction: "mean" | "sum" | "none"

    Returns:
        scalar Tensor (or unreduced Tensor if reduction="none")
    """
    return F.mse_loss(pred, target, reduction=reduction)


def ce_loss(
    logits: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Cross-Entropy loss for classification.

    Args:
        logits: Tensor[B, num_classes]
        target: Tensor[B] (class indices)
        reduction: "mean" | "sum" | "none"

    Returns:
        scalar Tensor (or unreduced Tensor if reduction="none")
    """
    return F.cross_entropy(logits, target, reduction=reduction)


def masked_reconstruction_loss(
    pred: Tensor,
    target: Tensor,
    recon_mask: Tensor,
    valid_mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Masked reconstruction loss.

    Args:
        pred: Tensor[B, C, T] or Tensor[B, N, D]
              Model reconstruction output
        target: Tensor[B, C, T] or Tensor[B, N, D]
                Ground-truth signal (clean or observed values)
        recon_mask: Tensor with same spatial shape as pred/target
                    1 = position masked for reconstruction
                    0 = position not used for reconstruction loss
        valid_mask: Optional Tensor with same spatial shape
                    1 = originally observed / valid
                    0 = originally missing
        reduction: "mean" | "sum"

    Returns:
        scalar Tensor
    """
    recon_mask = recon_mask.float()

    if valid_mask is not None:
        valid_mask = valid_mask.float()
        mask = recon_mask * valid_mask
    else:
        mask = recon_mask

    loss = (pred - target) ** 2

    loss = loss * mask

    if reduction == "sum":
        return loss.sum()

    elif reduction == "mean":
        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom

    else:
        raise ValueError(f"Unknown reduction: {reduction}")
