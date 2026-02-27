"""
InfSA Attention Module — Drop-in nn.Module Replacement
=======================================================

Provides ``InfSAAttention``, an ``nn.Module`` that is API-compatible with
``nn.MultiheadAttention`` while using InfSA attention internally.

Usage::

    # Drop-in replacement for nn.MultiheadAttention
    attn = InfSAAttention(embed_dim=768, num_heads=12, variant="pure_infsa")
    output, weights = attn(query, key, value)

The module handles Q/K/V projection, multi-head splitting, InfSA score
computation, and output projection — all in one self-contained module.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from infsa.core import infsa_attention


class InfSAAttention(nn.Module):
    """Multi-Head Infinite Self-Attention module.

    Drop-in replacement for ``nn.MultiheadAttention`` that uses InfSA
    instead of softmax attention.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of parallel attention heads.
        variant: InfSA variant — ``"pure_infsa"`` or ``"linear_infsa"``.
        dropout: Dropout probability on attention scores. Default: 0.0.
        bias: If True, add bias to input/output projections. Default: True.
        batch_first: If True, input/output shape is ``(B, N, E)``.
            If False, shape is ``(N, B, E)``. Default: True.
        rho_init: Initial value for the learnable ρ parameter. Default: 0.95.
            Internally stored as a logit: ρ = sigmoid(rho_logit).
        rho_trainable: If True, ρ is a learnable parameter. Default: True.

    Shape:
        - query: ``(B, N, E)`` if batch_first else ``(N, B, E)``
        - key: ``(B, S, E)`` if batch_first else ``(S, B, E)``
        - value: ``(B, S, E)`` if batch_first else ``(S, B, E)``
        - Output: ``(B, N, E)`` if batch_first else ``(N, B, E)``

    Example::

        >>> attn = InfSAAttention(embed_dim=512, num_heads=8, variant="pure_infsa")
        >>> x = torch.randn(4, 197, 512)  # (B, N, E)
        >>> out, scores = attn(x, x, x)
        >>> out.shape
        torch.Size([4, 197, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        variant: str = "pure_infsa",
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        rho_init: float = 0.95,
        rho_trainable: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        super().__init__()
        if variant not in ("pure_infsa", "linear_infsa"):
            raise ValueError(
                f"Unknown variant '{variant}'. Choose 'pure_infsa' or 'linear_infsa'."
            )
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be > 0, got {embed_dim} and {num_heads}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.variant = variant
        self.dropout = dropout
        self.batch_first = batch_first

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Trainable rho via logit reparameterization
        rho_logit = math.log(rho_init / (1.0 - rho_init))
        if rho_trainable:
            self.rho_logit = nn.Parameter(torch.tensor(rho_logit))
        else:
            self.register_buffer("rho_logit", torch.tensor(rho_logit))

        self._reset_parameters()

    @property
    def rho(self) -> float:
        """Current ρ value (sigmoid of the stored logit)."""
        return torch.sigmoid(self.rho_logit).item()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: Optional[Tensor] = None,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            key_padding_mask: Not used by InfSA (accepted for API compatibility).
            need_weights: If True, return attention scores.
            attn_mask: Not used by InfSA (accepted for API compatibility).
            average_attn_weights: If True, average weights across heads.
            is_causal: Not used by InfSA (accepted for API compatibility).

        Returns:
            Tuple of (output, attention_weights). Weights are None if
            need_weights is False.
        """
        # HuggingFace compatibility: accept hidden_states as self-attention input
        hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is not None and query is None:
            query = key = value = hidden_states

        # Handle seq-first format
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, N, _ = query.shape
        S = key.shape[1]
        H = self.num_heads
        D = self.head_dim

        # Project Q, K, V
        q = self.q_proj(query).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        k = self.k_proj(key).view(B, S, H, D).transpose(1, 2)    # (B, H, S, D)
        v = self.v_proj(value).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)

        # Get rho from learnable parameter
        rho_val = torch.sigmoid(self.rho_logit).item()

        # Compute InfSA attention
        output = infsa_attention(
            q, k, v,
            variant=self.variant,
            rho=rho_val,
            dropout_p=self.dropout if self.training else 0.0,
            training=self.training,
        )  # (B, H, N, D)

        # Collect attention weights if requested
        attn_weights = None
        if need_weights:
            from infsa.core import pure_infsa_scores, linear_infsa_scores
            if self.variant == "pure_infsa":
                attn_weights = pure_infsa_scores(q, k, rho=rho_val)
            else:
                attn_weights = linear_infsa_scores(q, k, rho=rho_val)
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)

        # Reshape back: (B, H, N, D) → (B, N, E)
        output = output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

        # Output projection
        output = self.out_proj(output)

        # Handle seq-first format
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, attn_weights
