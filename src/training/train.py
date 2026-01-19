from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.models.transformer import SpectralTransformerEncoder
from src.training.losses import (
    mse_loss,
    ce_loss,
    masked_reconstruction_loss,
)


class SupervisedModel(nn.Module):
    """
    Wrapper: Transformer encoder + supervised head.
    """

    def __init__(
        self,
        encoder: SpectralTransformerEncoder,
        task: str,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        assert task in {"classification", "regression"}
        self.task = task

        self.encoder = encoder

        if task == "classification":
            assert num_classes is not None
            self.head = nn.Linear(encoder.norm.normalized_shape[0], num_classes)
        else:
            self.head = nn.Linear(encoder.norm.normalized_shape[0], 1)

    def forward(self, x, mask=None, q=None):
        _, pooled = self.encoder(x, mask=mask, q=q)
        out = self.head(pooled)

        if self.task == "regression":
            out = out.squeeze(-1)

        return out


class ReconstructionHead(nn.Module):
    """
    Simple linear reconstruction head for MAE-style pretraining.
    """

    def __init__(self, d_model: int, n_channels: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_channels)

    def forward(self, tokens):
        return self.proj(tokens)


def train_step(
    model: SupervisedModel,
    recon_head: Optional[ReconstructionHead],
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    task: str,
    recon_weight: float = 0.0,
    device: str = "cpu",
):
    model.train()
    optimizer.zero_grad()

    x = batch["x"].to(device)
    mask = batch["mask"].to(device)

    if task == "classification":
        y = batch["y"].to(device)
    else:
        y = batch["y"].float().to(device)

    logits = model(x, mask=mask)

    if task == "classification":
        loss_sup = ce_loss(logits, y)
    else:
        loss_sup = mse_loss(logits, y)

    loss = loss_sup

    if recon_head is not None and recon_weight > 0.0:
        tokens, _ = model.encoder(x, mask=mask)

        recon_mask = batch["recon_mask"].to(device)
        valid_mask = batch["mask"].to(device)

        recon_pred = recon_head(tokens).transpose(1, 2)
        recon_target = batch["x_clean"].to(device)

        loss_recon = masked_reconstruction_loss(
            pred=recon_pred,
            target=recon_target,
            recon_mask=recon_mask,
            valid_mask=valid_mask,
        )

        loss = loss + recon_weight * loss_recon

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "loss_sup": loss_sup.item(),
    }


@torch.no_grad()
def eval_step(
    model: SupervisedModel,
    loader: DataLoader,
    task: str,
    device: str = "cpu",
):
    model.eval()

    total = 0
    correct = 0
    se = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)

        y = batch["y"].to(device)

        out = model(x, mask=mask)

        if task == "classification":
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        else:
            se += ((out - y.float()) ** 2).sum().item()
            total += y.numel()

    if task == "classification":
        return {"accuracy": correct / max(total, 1)}
    else:
        return {"rmse": (se / max(total, 1)) ** 0.5}


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
):
    device = cfg.get("device", "cpu")

    encoder = SpectralTransformerEncoder(
        n_channels=cfg["data"]["channels"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        attn_mode=cfg["model"]["attn_mode"],
        d_q=None,
        pooling=cfg["model"].get("pooling", "mean"),
    )

    model = SupervisedModel(
        encoder=encoder,
        task=cfg["task"]["type"],
        num_classes=cfg["task"].get("num_classes"),
    ).to(device)

    recon_head = None
    if cfg["training"].get("use_reconstruction", False):
        recon_head = ReconstructionHead(
            d_model=cfg["model"]["d_model"],
            n_channels=cfg["data"]["channels"],
        ).to(device)

    params = list(model.parameters())
    if recon_head is not None:
        params += list(recon_head.parameters())

    optimizer = AdamW(
        params,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-2),
    )

    for epoch in range(cfg["training"]["epochs"]):
        for batch in train_loader:
            stats = train_step(
                model=model,
                recon_head=recon_head,
                batch=batch,
                optimizer=optimizer,
                task=cfg["task"]["type"],
                recon_weight=cfg["training"].get("recon_weight", 0.0),
                device=device,
            )

        val_stats = eval_step(
            model=model,
            loader=val_loader,
            task=cfg["task"]["type"],
            device=device,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={stats['loss']:.4f} | "
            f"val={val_stats}"
        )

    return model
