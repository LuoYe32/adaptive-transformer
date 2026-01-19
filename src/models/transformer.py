from typing import Optional, Literal

import torch
import torch.nn as nn
from torch import Tensor

from src.models.attention import MultiHeadAttentionIFA


class TimeTokenEmbedding(nn.Module):
    """
    Simple tokenization: treat each time step as a token.

    Input:
        x: Tensor[B, C, T]

    Output:
        tokens: Tensor[B, T, d_model]
        mask: Tensor[B, T]  (1 = observed, 0 = missing)
    """

    def __init__(self, n_channels: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_channels, d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        tokens = self.proj(x)

        if mask is not None:
            token_mask = (mask.sum(dim=1) > 0).float()
        else:
            token_mask = None

        return tokens, token_mask


class PatchEmbedding1D(nn.Module):
    """
    Patch-based tokenization for long 1D signals.

    Splits the time axis into non-overlapping patches.

    Input:
        x: Tensor[B, C, T]

    Output:
        tokens: Tensor[B, N_patches, d_model]
        mask: Tensor[B, N_patches]
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        patch_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(n_channels * patch_size, d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        B, C, T = x.shape
        P = self.patch_size
        assert T % P == 0, "T must be divisible by patch_size"

        N = T // P

        x = x.view(B, C, N, P)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, N, C * P)

        tokens = self.proj(x)

        if mask is not None:
            mask = mask.view(B, C, N, P)
            patch_mask = (mask.sum(dim=(1, 3)) > 0).float()
        else:
            patch_mask = None

        return tokens, patch_mask


class FeedForward(nn.Module):
    """
    Standard Transformer feed-forward network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    Vanilla Transformer encoder block with IFA-compatible attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attn_mode: Literal["vanilla", "mask_token", "observed_only"],
        d_q: Optional[int] = None,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = MultiHeadAttentionIFA(
            d_model=d_model,
            n_heads=n_heads,
            attn_mode=attn_mode,
            d_q=d_q,
            dropout=dropout,
        )

        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, q=q)
        x = x + self.ffn(self.norm2(x))
        return x


class SpectralTransformerEncoder(nn.Module):
    """
    Transformer encoder for multichannel / spectral data.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        attn_mode: Literal["vanilla", "mask_token", "observed_only"] = "vanilla",
        d_q: Optional[int] = None,
        tokenization: Literal["time", "patch"] = "time",
        patch_size: Optional[int] = None,
        max_tokens: int = 2048,
        pooling: Literal["mean", "cls"] = "mean",
    ):
        """
        Args:
            n_channels: number of input channels
            d_model: transformer model dimension
            n_layers: number of encoder layers
            n_heads: number of attention heads
            d_ff: feed-forward dimension
            dropout: dropout probability
            attn_mode: attention mode ("vanilla", "mask_token", "observed_only")
            d_q: dimension of quality embedding (None = ignore q)
            tokenization: "time" or "patch"
            patch_size: patch size (required if tokenization == "patch")
            max_tokens: maximum number of tokens for positional embeddings
            pooling: "mean" or "cls"
        """
        super().__init__()

        self.pooling = pooling

        if tokenization == "time":
            self.tokenizer = TimeTokenEmbedding(n_channels, d_model)
        elif tokenization == "patch":
            assert patch_size is not None
            self.tokenizer = PatchEmbedding1D(
                n_channels, d_model, patch_size
            )
        else:
            raise ValueError(f"Unknown tokenization: {tokenization}")

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_tokens + 1, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_mode=attn_mode,
                    d_q=d_q,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
    ):
        """
        Args:
            x: Tensor[B, C, T]
            mask: Tensor[B, C, T] or None
            q: Tensor[B, d_q] or None

        Returns:
            encoder_out: Tensor[B, N, d_model]
            pooled: Tensor[B, d_model]
        """
        tokens, token_mask = self.tokenizer(x, mask)

        B, N, _ = tokens.shape

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

            if token_mask is not None:
                cls_mask = torch.ones(B, 1, device=token_mask.device)
                token_mask = torch.cat([cls_mask, token_mask], dim=1)

        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        tokens = self.dropout(tokens)

        for layer in self.layers:
            tokens = layer(tokens, mask=token_mask, q=q)

        tokens = self.norm(tokens)

        if self.pooling == "cls":
            pooled = tokens[:, 0]
        else:
            pooled = tokens.mean(dim=1)

        return tokens, pooled
