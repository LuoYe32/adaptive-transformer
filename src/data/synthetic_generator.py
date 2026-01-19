from typing import Dict, Tuple, Optional
import math
import random

import torch
from torch import Tensor


def _sample_cfg(cfg: dict) -> dict:
    """
    Sample a concrete configuration from a possibly list-valued config.
    """
    sampled = {}
    for k, v in cfg.items():
        if isinstance(v, list):
            sampled[k] = random.choice(v)
        else:
            sampled[k] = v
    return sampled


class SyntheticSpectralGenerator:
    """
    Generator for synthetic multichannel spectral data.

    Each sample consists of:
        - clean signal
        - corrupted signal
        - missingness mask
        - metadata describing corruption parameters
    """

    def __init__(
        self,
        n_channels: int,
        length: int,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_channels: number of spectral channels
            length: length of each channel
            device: torch device
            seed: random seed for reproducibility
        """
        self.n_channels = n_channels
        self.length = length
        self.device = device or torch.device("cpu")

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

    def generate_clean(self) -> Tensor:
        """
        Generate a clean multichannel spectral signal.

        Uses a mixture of smooth components:
            - sinusoidal bases
            - Gaussian peaks
            - low-frequency trends

        Returns:
            Tensor[C, T]
        """
        t = torch.linspace(0, 1, self.length, device=self.device)
        x = torch.zeros(self.n_channels, self.length, device=self.device)

        for c in range(self.n_channels):
            freq = random.uniform(1.0, 5.0)
            phase = random.uniform(0, 2 * math.pi)
            signal = torch.sin(2 * math.pi * freq * t + phase)

            n_peaks = random.randint(1, 3)
            for _ in range(n_peaks):
                center = random.uniform(0.1, 0.9)
                width = random.uniform(0.02, 0.08)
                amplitude = random.uniform(0.5, 1.5)
                signal += amplitude * torch.exp(
                    -0.5 * ((t - center) / width) ** 2
                )

            scale = random.uniform(0.5, 2.0)
            x[c] = scale * signal

        return x

    def apply_noise(self, x: Tensor, noise_cfg: Dict) -> Tensor:
        """
        Apply noise corruption to a clean signal.

        Supported noise types:
            - Gaussian
            - Poisson
            - multiplicative channel noise

        Args:
            x: Tensor[C, T]
            noise_cfg: dictionary describing noise parameters

        Returns:
            Tensor[C, T]
        """
        x_noisy = x.clone()

        noise_type = noise_cfg.get("type", "gaussian")

        if noise_type == "gaussian":
            sigma = noise_cfg.get("sigma", 0.1)
            noise = torch.randn_like(x_noisy) * sigma
            x_noisy = x_noisy + noise

        elif noise_type == "poisson":
            lam = noise_cfg.get("lambda", 20.0)
            x_shifted = x_noisy - x_noisy.min()
            noisy = torch.poisson(x_shifted * lam) / lam
            x_noisy = noisy

        elif noise_type == "multiplicative":
            sigma = noise_cfg.get("sigma", 0.1)
            scale = 1.0 + torch.randn(self.n_channels, 1, device=x.device) * sigma
            x_noisy = x_noisy * scale

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return x_noisy

    @staticmethod
    def apply_missingness(x: Tensor, pattern_cfg: Dict) -> Tuple[Tensor, Tensor]:
        """
        Apply missingness pattern to a signal.

        Supported patterns:
            - random
            - block
            - line
            - channel_dropout

        Args:
            x: Tensor[C, T]
            pattern_cfg: dictionary describing missingness parameters

        Returns:
            x_corrupted: Tensor[C, T]
            mask: Tensor[C, T] (1 = observed, 0 = missing)
        """
        C, T = x.shape
        mask = torch.ones_like(x)

        pattern = pattern_cfg.get("type", "random")
        frac = pattern_cfg.get("fraction", 0.1)
        frac = min(max(frac, 0.0), 1.0)

        if pattern == "random":
            missing = torch.rand_like(x) < frac
            mask[missing] = 0.0

        elif pattern == "block":
            block_size = max(1, int(T * frac))
            start = random.randint(0, max(0, T - block_size))
            mask[:, start : start + block_size] = 0.0

        elif pattern == "line":
            center = random.randint(0, T - 1)
            width = max(1, int(T * frac * 0.2))
            left = max(0, center - width)
            right = min(T, center + width)
            mask[:, left:right] = 0.0

        elif pattern == "channel_dropout":
            n_drop = max(1, int(C * frac))
            channels = random.sample(range(C), n_drop)
            for c in channels:
                mask[c, :] = 0.0

        else:
            raise ValueError(f"Unknown missingness pattern: {pattern}")

        x_corrupted = x * mask
        return x_corrupted, mask

    def generate_sample(self, cfg: Dict) -> Dict:
        """
        Generate a full synthetic sample.

        Pipeline:
            clean -> noise -> missingness

        Args:
            cfg: configuration dictionary with keys:
                - noise
                - missing

        Returns:
            dict with keys:
                - x_clean
                - x
                - mask
                - meta
        """
        x_clean = self.generate_clean()

        raw_noise_cfg = cfg.get("noise", {})
        raw_missing_cfg = cfg.get("missing", {})

        noise_cfg = _sample_cfg(raw_noise_cfg)
        missing_cfg = _sample_cfg(raw_missing_cfg)

        x_noisy = self.apply_noise(x_clean, noise_cfg)
        x_corrupted, mask = self.apply_missingness(x_noisy, missing_cfg)

        meta = {
            "noise": noise_cfg,
            "missing": missing_cfg,
            "missing_fraction": float(mask.numel() - mask.sum()) / mask.numel(),
        }

        return {
            "x_clean": x_clean,
            "x": x_corrupted,
            "mask": mask,
            "meta": meta,
        }
