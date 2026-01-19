from torch import Tensor
import torch


def _safe_median(x: Tensor, mask: Tensor) -> Tensor:
    """
    Compute masked median along the last dimension.

    Returns:
        Tensor[B, C]
    """
    x_masked = x.clone()
    x_masked[mask == 0] = float("nan")
    return torch.nanmedian(x_masked, dim=-1).values


def _safe_mean(x: Tensor, mask: Tensor) -> Tensor:
    """
    Compute masked mean along the last dimension.

    Returns:
        Tensor[B, C]
    """
    masked_sum = (x * mask).sum(dim=-1)
    count = mask.sum(dim=-1).clamp(min=1.0)
    return masked_sum / count


def _safe_var(x: Tensor, mask: Tensor) -> Tensor:
    """
    Compute masked variance along the last dimension.

    Returns:
        Tensor[B, C]
    """
    mean = _safe_mean(x, mask)
    diff2 = (x - mean.unsqueeze(-1)) ** 2
    var = (diff2 * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    return var


def fraction_missing(mask: Tensor) -> Tensor:
    """
    Fraction of missing values per channel.

    Args:
        mask: Tensor[B, C, T], 1 = observed, 0 = missing

    Returns:
        Tensor[B, C]
    """
    total = mask.shape[-1]
    observed = mask.sum(dim=-1)
    return 1.0 - (observed / total)


def snr_mad(x: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Robust signal-to-noise ratio estimate using Median Absolute Deviation (MAD).

    SNR ≈ |median(x)| / MAD(x)

    Args:
        x: Tensor[B, C, T]
        mask: Tensor[B, C, T]
        eps: numerical stability constant

    Returns:
        Tensor[B, C]
    """
    median = _safe_median(x, mask)

    abs_dev = torch.abs(x - median.unsqueeze(-1))
    abs_dev[mask == 0] = float("nan")
    mad = torch.nanmedian(abs_dev, dim=-1).values

    return torch.abs(median) / (mad + eps)


def local_variance(x: Tensor, mask: Tensor, window: int = 5) -> Tensor:
    """
    Local (smoothed) variance estimate per channel.

    Uses a sliding window to estimate local variability and then averages it.

    Args:
        x: Tensor[B, C, T]
        mask: Tensor[B, C, T]
        window: size of sliding window

    Returns:
        Tensor[B, C]
    """
    assert window >= 2, "window size must be >= 2"

    B, C, T = x.shape
    pad = window // 2

    x_padded = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
    mask_padded = torch.nn.functional.pad(mask, (pad, pad), mode="constant", value=0)

    local_vars = []

    for i in range(T):
        x_win = x_padded[..., i : i + window]
        m_win = mask_padded[..., i : i + window]

        var = _safe_var(x_win, m_win)
        local_vars.append(var)

    local_vars = torch.stack(local_vars, dim=-1)  # [B, C, T]
    return local_vars.mean(dim=-1)


def kurtosis(x: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Kurtosis per channel (Fisher definition, without subtracting 3).

    Measures tail-heaviness / peakedness of the distribution.

    Args:
        x: Tensor[B, C, T]
        mask: Tensor[B, C, T]

    Returns:
        Tensor[B, C]
    """
    mean = _safe_mean(x, mask)
    var = _safe_var(x, mask) + eps

    centered = x - mean.unsqueeze(-1)
    centered[mask == 0] = 0.0

    m4 = ((centered ** 4) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    return m4 / (var ** 2)


def fraction_outliers(x: Tensor, mask: Tensor, thresh: float = 3.5) -> Tensor:
    """
    Fraction of outliers per channel using robust z-score (MAD-based).

    z = 0.6745 * (x - median) / MAD

    Args:
        x: Tensor[B, C, T]
        mask: Tensor[B, C, T]
        thresh: threshold on |z| to count outliers

    Returns:
        Tensor[B, C]
    """
    median = _safe_median(x, mask)

    abs_dev = torch.abs(x - median.unsqueeze(-1))
    abs_dev[mask == 0] = float("nan")
    mad = torch.nanmedian(abs_dev, dim=-1).values + 1e-6

    z = 0.6745 * (x - median.unsqueeze(-1)) / mad.unsqueeze(-1)
    z = torch.abs(z) * mask

    outliers = (z > thresh).float().sum(dim=-1)
    total = mask.sum(dim=-1).clamp(min=1.0)

    return outliers / total


def global_quality_summary(per_channel_feats: Tensor) -> Tensor:
    """
    Aggregate per-channel quality features into global summary statistics.

    Computes mean, std, and max over channels.

    Args:
        per_channel_feats: Tensor[B, C, F]

    Returns:
        Tensor[B, 3 * F]
    """
    mean = per_channel_feats.mean(dim=1)
    std = per_channel_feats.std(dim=1)
    maxv = per_channel_feats.max(dim=1).values

    return torch.cat([mean, std, maxv], dim=-1)
