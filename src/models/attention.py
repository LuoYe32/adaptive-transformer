from typing import Optional
import math

import torch
import torch.nn as nn
from torch import Tensor


def _reshape_heads(x: Tensor, n_heads: int) -> Tensor:
    """
    Reshape (B, N, d_model) -> (B, n_heads, N, d_head)
    """
    B, N, D = x.shape
    d_head = D // n_heads
    x = x.view(B, N, n_heads, d_head)
    return x.transpose(1, 2)


def _merge_heads(x: Tensor) -> Tensor:
    """
    Reshape (B, n_heads, N, d_head) -> (B, N, d_model)
    """
    B, H, N, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, N, H * Dh)


class MultiHeadAttentionIFA(nn.Module):
    """
    Multi-Head Self-Attention with Imputation-Free Attention (IFA).

    Supported modes:
        - "vanilla": standard attention with masking
        - "mask_token": missing values are replaced by a learned mask token
        - "observed_only": attention does not attend to missing keys/values

    Optionally conditioned on a global quality embedding q.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_mode: str = "vanilla",
        d_q: Optional[int] = None,
        dropout: float = 0.1,
        use_uncertainty_bias: bool = False,
    ):
        """
        Args:
            d_model: model dimension
            n_heads: number of attention heads
            attn_mode: "vanilla", "mask_token", or "observed_only"
            d_q: dimension of quality embedding q (if None, q is ignored)
            dropout: attention dropout
            use_uncertainty_bias: whether to add a learned bias for mask tokens
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert attn_mode in {"vanilla", "mask_token", "observed_only"}

        self.d_model = d_model
        self.n_heads = n_heads
        self.attn_mode = attn_mode
        self.use_uncertainty_bias = use_uncertainty_bias

        self.scale = math.sqrt(d_model // n_heads)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if attn_mode == "mask_token":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.mask_token, std=0.02)

        if use_uncertainty_bias:
            self.uncertainty_bias = nn.Parameter(torch.zeros(n_heads, 1, 1))

        if d_q is not None:
            self.use_quality = True
            self.q_to_heads = nn.Linear(d_q, n_heads)
        else:
            self.use_quality = False


    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Tensor[B, N, d_model]         -- input tokens
            mask: Tensor[B, N] or None       -- 1 = observed, 0 = missing
            q: Tensor[B, d_q] or None        -- global quality embedding

        Returns:
            Tensor[B, N, d_model]
        """
        B, N, _ = x.shape

        if self.attn_mode == "mask_token" and mask is not None:
            mask_exp = mask.unsqueeze(-1)  # (B, N, 1)
            x = x * mask_exp + self.mask_token * (1.0 - mask_exp)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = _reshape_heads(Q, self.n_heads)
        K = _reshape_heads(K, self.n_heads)
        V = _reshape_heads(V, self.n_heads)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if self.attn_mode == "observed_only" and mask is not None:
            # mask_k: (B, 1, 1, N)
            mask_k = mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(mask_k == 0, float("-inf"))

        if self.use_uncertainty_bias and mask is not None:
            mask_k = mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits + self.uncertainty_bias * (1.0 - mask_k)

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # (B, H, N, d_head)

        if self.use_quality and q is not None:
            # gate: (B, H) -> (B, H, 1, 1)
            gate = torch.sigmoid(self.q_to_heads(q)).unsqueeze(-1).unsqueeze(-1)
            out = out * gate

        out = _merge_heads(out)
        out = self.out_proj(out)

        return out
