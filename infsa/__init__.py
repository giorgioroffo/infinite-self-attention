"""
InfSA — Infinite Self-Attention
================================

A drop-in replacement for standard self-attention in any Transformer model.

InfSA replaces the softmax(QK^T/√d) attention mechanism with spectral
approximation techniques inspired by Infinite Feature Selection (Inf-FS),
achieving efficient global context modeling.

Available Variants:
    - ``pure_infsa``: Single-hop spectral attention A = ρ · Norm(ReLU(QK^T)).
      Multi-hop behavior emerges through stacked Transformer layers.
    - ``linear_infsa``: Eigenvector approximation via power iteration.
      O(N·D) complexity — avoids building the full N×N attention matrix.

Quick Start::

    import infsa

    # Replace all attention layers in a model
    model = infsa.convert(model, variant="pure_infsa")

    # Or use the functional API directly
    attn_out = infsa.infsa_attention(q, k, v, variant="pure_infsa")

    # Or use the nn.Module drop-in replacement
    attn = infsa.InfSAAttention(embed_dim=768, num_heads=12, variant="pure_infsa")

Reference:
    Roffo et al., "Infinite Self-Attention", 2025.
    Inspired by: Roffo et al., "Infinite Feature Selection" (ICCV 2015, TPAMI 2020).
"""

__version__ = "0.1.0"

from infsa.core import (
    pure_infsa_scores,
    linear_infsa_scores,
    infsa_attention,
)
from infsa.attention import InfSAAttention
from infsa.convert import convert, replace_attention

__all__ = [
    # Functional API
    "pure_infsa_scores",
    "linear_infsa_scores",
    "infsa_attention",
    # Module API
    "InfSAAttention",
    # Conversion API
    "convert",
    "replace_attention",
]
