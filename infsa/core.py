"""
InfSA Core — Functional Attention Score Computation
====================================================

This module provides the pure functional implementations of InfSA attention
score computation. These functions operate on pre-projected Q, K, V tensors
and return attention outputs, making them easy to integrate into any
Transformer architecture.

Two variants are provided:

1. **pure_infsa** — Spectral single-hop attention
   Computes A = ρ · Norm(ReLU(QK^T)) where Norm is Frobenius normalization.
   Multi-hop behavior (ρA + ρ²A² + ...) emerges through stacked layers.

2. **linear_infsa** — Eigenvector approximation via power iteration
   Approximates the principal eigenvector of the implicit attention matrix
   A = ReLU(QK^T) using O(N·D) operations per iteration, avoiding the
   explicit N×N matrix construction.

Reference:
    Roffo et al., "Infinite Self-Attention", 2025.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def pure_infsa_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    rho: float = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute pure InfSA attention scores.

    Builds the single-hop attention matrix:
        A = ρ · Norm(ReLU(QK^T / d))

    where Norm(·) is Frobenius normalization ensuring ||A||_F < 1 for
    guaranteed convergence of the infinite series across layers.

    Args:
        q: Query tensor of shape ``(B, H, N, D)`` or ``(B*H, N, D)``.
        k: Key tensor of shape ``(B, H, N, D)`` or ``(B*H, N, D)``.
        rho: Decay parameter in (0, 1). Controls the spectral radius of A.
        eps: Small constant for numerical stability.

    Returns:
        Attention score matrix of shape ``(B, H, N, N)`` or ``(B*H, N, N)``.

    Example::

        >>> q = torch.randn(2, 8, 197, 64)  # (B, H, N, D)
        >>> k = torch.randn(2, 8, 197, 64)
        >>> scores = pure_infsa_scores(q, k, rho=0.9)
        >>> scores.shape
        torch.Size([2, 8, 197, 197])
    """
    D = q.shape[-1]
    scale = math.sqrt(1.0 / D)
    q_scaled = q * scale
    k_scaled = k * scale

    # Affinity matrix A = ReLU(QK^T / d)
    A = torch.matmul(q_scaled, k_scaled.transpose(-2, -1))  # (..., N, N)
    A = torch.relu(A)

    # Frobenius normalization over the last two dims
    # Flatten last two dims for norm computation, then reshape
    orig_shape = A.shape
    A_flat = A.flatten(-2)  # (..., N*N)
    frob_norm = torch.norm(A_flat, p=2, dim=-1, keepdim=True)  # (..., 1)
    A_flat = A_flat / (frob_norm + eps)
    A = A_flat.view(orig_shape)

    return rho * A


def linear_infsa_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    rho: float = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute linear InfSA attention scores via eigenvector approximation.

    Approximates the principal eigenvector of the implicit attention matrix
    A = ReLU(QK^T) using a soft query pooling + single matrix-vector product,
    yielding O(N·D) complexity instead of O(N²).

    The algorithm:
        1. Compute energy-based query summary: q̄ = Σ_t w_t · q_t
        2. Compute attention: a = normalize(ReLU(q̄ · K^T / d))
        3. Return: ρ · a  (shape: per-token importance scores)

    The returned scores are per-token importance weights (not a full N×N
    matrix), which are used to compute a weighted sum of values.

    Args:
        q: Query tensor of shape ``(B, H, N, D)`` or ``(B*H, N, D)``.
        k: Key tensor of shape ``(B, H, N, D)`` or ``(B*H, N, D)``.
        rho: Decay parameter in (0, 1).
        eps: Small constant for numerical stability.

    Returns:
        Attention importance vector of shape ``(..., N, 1)``, broadcastable
        for element-wise multiplication with values.

    Example::

        >>> q = torch.randn(2, 8, 197, 64)  # (B, H, N, D)
        >>> k = torch.randn(2, 8, 197, 64)
        >>> scores = linear_infsa_scores(q, k, rho=0.9)
        >>> scores.shape
        torch.Size([2, 8, 197, 1])
    """
    D = q.shape[-1]
    scale = math.sqrt(1.0 / D)
    q_scaled = q * scale
    k_scaled = k * scale

    # Step 1: Soft energy-based query pooling → q̄
    energy = torch.relu(q_scaled.norm(p=2, dim=-1))  # (..., N)
    energy_sum = energy.sum(dim=-1, keepdim=True) + eps  # (..., 1)
    weights = energy / energy_sum  # (..., N)
    q_bar = torch.einsum("...n,...nd->...d", weights, q_scaled)  # (..., D)

    # Step 2: Compute attention scores a = normalize(ReLU(q̄ · K^T))
    a = torch.relu(torch.einsum("...d,...nd->...n", q_bar, k_scaled))  # (..., N)
    a = a / (a.sum(dim=-1, keepdim=True) + eps)

    # Step 3: Apply decay and return as column vector for broadcasting
    return (rho * a).unsqueeze(-1)  # (..., N, 1)


def infsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variant: str = "pure_infsa",
    rho: float = 0.95,
    dropout_p: float = 0.0,
    training: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute InfSA attention output from pre-projected Q, K, V.

    This is the main functional API. It computes attention scores using the
    specified InfSA variant and applies them to values.

    For ``pure_infsa``:
        output = A @ V  where A = ρ · Norm(ReLU(QK^T / d))

    For ``linear_infsa``:
        output = broadcast(scores) * V  (element-wise, then summed)
        where scores are per-token importance weights.

    Args:
        q: Query tensor ``(B, H, N, D)`` or ``(B*H, N, D)``.
        k: Key tensor ``(B, H, N, D)`` or ``(B*H, N, D)``.
        v: Value tensor ``(B, H, N, D)`` or ``(B*H, N, D)``.
        variant: Which InfSA variant: ``"pure_infsa"`` or ``"linear_infsa"``.
        rho: Decay parameter in (0, 1).
        dropout_p: Dropout probability on attention scores (training only).
        training: Whether the model is in training mode.
        eps: Numerical stability constant.

    Returns:
        Attention output tensor, same shape as ``v``.

    Example::

        >>> q = torch.randn(2, 8, 197, 64)
        >>> k = torch.randn(2, 8, 197, 64)
        >>> v = torch.randn(2, 8, 197, 64)
        >>> out = infsa_attention(q, k, v, variant="pure_infsa", rho=0.9)
        >>> out.shape
        torch.Size([2, 8, 197, 64])
    """
    if variant == "pure_infsa":
        scores = pure_infsa_scores(q, k, rho=rho, eps=eps)  # (..., N, N)
        if dropout_p > 0.0 and training:
            scores = F.dropout(scores, p=dropout_p)
        return torch.matmul(scores, v)

    elif variant == "linear_infsa":
        scores = linear_infsa_scores(q, k, rho=rho, eps=eps)  # (..., N, 1)
        if dropout_p > 0.0 and training:
            scores = F.dropout(scores, p=dropout_p)
        # Weighted sum: each token gets the globally-pooled representation
        pooled = (scores * v).sum(dim=-2, keepdim=True)  # (..., 1, D)
        N = v.shape[-2]
        return pooled.expand_as(v)  # (..., N, D) — broadcast to all positions

    else:
        raise ValueError(
            f"Unknown InfSA variant: '{variant}'. "
            f"Choose from: 'pure_infsa', 'linear_infsa'."
        )
