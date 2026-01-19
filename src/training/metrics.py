import torch
from torch import Tensor
from typing import Dict, List


def accuracy(pred: Tensor, target: Tensor) -> float:
    """
    Compute classification accuracy.
    """
    return (pred == target).float().mean().item()


def rmse(pred: Tensor, target: Tensor) -> float:
    """
    Root Mean Squared Error.
    """
    return torch.sqrt(((pred - target) ** 2).mean()).item()


@torch.no_grad()
def accuracy_vs_missing_fraction(
    model,
    loader,
    device: str = "cpu",
) -> Dict[float, float]:
    """
    Compute accuracy grouped by missing_fraction.

    Returns:
        dict: {missing_fraction -> accuracy}
    """
    model.eval()

    stats: Dict[float, List[int]] = {}

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)

        missing_frac = batch["meta"]["missing_fraction"]

        logits = model(x, mask=mask)
        preds = logits.argmax(dim=1)

        for i, mf in enumerate(missing_frac):
            mf = float(mf)
            if mf not in stats:
                stats[mf] = [0, 0]

            stats[mf][0] += int(preds[i] == y[i])
            stats[mf][1] += 1

    return {mf: c / t for mf, (c, t) in stats.items()}
