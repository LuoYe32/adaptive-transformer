from typing import Dict, Any
import torch
from torch import Tensor

from src.data.synthetic_generator import SyntheticSpectralGenerator


class SyntheticSpectralDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for synthetic spectral data.

    Each item contains:
        - corrupted input x
        - missingness mask
        - clean reference signal
        - metadata describing the corruption
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Args:
            cfg: configuration dictionary with keys:
                data:
                    - channels
                    - length
                generator:
                    - n_samples
                    - seed (optional)
                noise:
                    - noise configuration (passed to generator)
                missing:
                    - missingness configuration (passed to generator)
        """
        self.cfg = cfg

        data_cfg = cfg["data"]
        gen_cfg = cfg.get("generator", {})
        self.n_samples = gen_cfg.get("n_samples", 1000)

        self.generator = SyntheticSpectralGenerator(
            n_channels=data_cfg["channels"],
            length=data_cfg["length"],
            seed=gen_cfg.get("seed", None),
        )

        self.noise_cfg = cfg.get("noise", {})
        self.missing_cfg = cfg.get("missing", {})

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Generate a single synthetic sample.

        Note:
            Samples are generated on-the-fly.
            Index is ignored except for reproducibility control.

        Returns:
            dict with keys:
                - x: Tensor[C, T]
                - mask: Tensor[C, T]
                - x_clean: Tensor[C, T]
                - meta: dict
        """
        sample = self.generator.generate_sample(
            {
                "noise": self.noise_cfg,
                "missing": self.missing_cfg,
            }
        )

        missing_fraction = sample["meta"]["missing_fraction"]
        y = int(missing_fraction > 0.2)

        return {
            "x": sample["x"].float(),
            "mask": sample["mask"].float(),
            "x_clean": sample["x_clean"].float(),
            "y": torch.tensor(y, dtype=torch.long),
            "meta": sample["meta"],
        }
