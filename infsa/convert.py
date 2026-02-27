"""
InfSA Convert — Automatic Model Conversion Utilities
=====================================================

Replace standard self-attention with InfSA in any PyTorch Transformer model.

The main entry point is :func:`convert`, which recursively walks a model's
module tree and replaces attention layers with InfSA equivalents.

Supported source attention types:
    - ``nn.MultiheadAttention`` (PyTorch built-in)
    - HuggingFace attention modules (auto-detected)
    - Any custom module matched by name or type

Usage::

    import infsa
    from torchvision.models import vit_b_16

    # One-liner: convert all attention in a ViT
    model = vit_b_16(weights="DEFAULT")
    model = infsa.convert(model, variant="pure_infsa")

    # Convert a HuggingFace model
    from transformers import AutoModel
    model = AutoModel.from_pretrained("bert-base-uncased")
    model = infsa.convert(model, variant="linear_infsa")
"""

import re
import copy
import logging
from typing import Optional, Set, Type, Callable, List, Tuple

import torch
import torch.nn as nn

from infsa.attention import InfSAAttention

logger = logging.getLogger(__name__)


def _get_embed_dim_and_heads(module: nn.Module) -> Tuple[Optional[int], Optional[int]]:
    """Extract embed_dim and num_heads from an attention module.

    Works with nn.MultiheadAttention and common HuggingFace attention patterns
    including LLaMA, Qwen, Mistral (GQA-style modules with head_dim).
    """
    # nn.MultiheadAttention
    if hasattr(module, "embed_dim") and hasattr(module, "num_heads"):
        return module.embed_dim, module.num_heads

    # HuggingFace: many attention modules store these as attributes
    embed_dim = None
    num_heads = None

    for attr in ("embed_dim", "hidden_size", "d_model", "embed_dimension", "all_head_size"):
        if hasattr(module, attr):
            embed_dim = getattr(module, attr)
            break

    for attr in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
        if hasattr(module, attr):
            num_heads = getattr(module, attr)
            break

    # Try to infer embed_dim from output projection (o_proj, out_proj, etc.)
    if embed_dim is None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name.lower() in ("o_proj", "out_proj", "c_proj"):
                embed_dim = child.out_features
                break
        # Fallback: any proj-like linear layer
        if embed_dim is None:
            for name, child in module.named_children():
                if isinstance(child, nn.Linear) and ("proj" in name.lower() or "out" in name.lower()):
                    embed_dim = child.out_features
                    break

    # Infer num_heads from head_dim (LLaMA, Qwen, Mistral, etc.)
    if num_heads is None and embed_dim is not None and hasattr(module, "head_dim"):
        head_dim = module.head_dim
        if head_dim > 0 and embed_dim % head_dim == 0:
            num_heads = embed_dim // head_dim

    return embed_dim, num_heads


def _is_attention_module(name: str, module: nn.Module) -> bool:
    """Check whether a module is likely an attention layer.

    Uses duck-typing: checks class name and typical attribute patterns.
    """
    # Direct match: PyTorch MHA
    if isinstance(module, nn.MultiheadAttention):
        return True

    # Class-name heuristics for HuggingFace and custom implementations
    cls_name = type(module).__name__.lower()
    attention_keywords = {"attention", "selfattention", "multiheadattention", "mha"}
    if any(kw in cls_name for kw in attention_keywords):
        # Additional check: must have projection-like children
        child_names = {n.lower() for n, _ in module.named_children()}
        has_proj = any(
            "proj" in n or "linear" in n or "dense" in n or "out" in n
            for n in child_names
        )
        has_params = sum(1 for _ in module.parameters()) > 0
        if has_proj or has_params:
            return True

    return False


def _convert_mha(
    module: nn.MultiheadAttention,
    variant: str,
    rho_init: float,
    rho_trainable: bool,
    copy_weights: bool,
) -> InfSAAttention:
    """Convert a nn.MultiheadAttention to InfSAAttention, optionally copying weights."""
    new_module = InfSAAttention(
        embed_dim=module.embed_dim,
        num_heads=module.num_heads,
        variant=variant,
        dropout=module.dropout,
        bias=module.in_proj_bias is not None,
        batch_first=module.batch_first,
        rho_init=rho_init,
        rho_trainable=rho_trainable,
        kdim=module.kdim,
        vdim=module.vdim,
    )

    if copy_weights:
        with torch.no_grad():
            # nn.MHA stores Q/K/V in a packed in_proj_weight (3*E, E)
            if module.in_proj_weight is not None:
                E = module.embed_dim
                q_w, k_w, v_w = module.in_proj_weight.chunk(3, dim=0)
                new_module.q_proj.weight.copy_(q_w)
                new_module.k_proj.weight.copy_(k_w)
                new_module.v_proj.weight.copy_(v_w)
            else:
                # Separate projection weights
                if module.q_proj_weight is not None:
                    new_module.q_proj.weight.copy_(module.q_proj_weight)
                if module.k_proj_weight is not None:
                    new_module.k_proj.weight.copy_(module.k_proj_weight)
                if module.v_proj_weight is not None:
                    new_module.v_proj.weight.copy_(module.v_proj_weight)

            if module.in_proj_bias is not None:
                E = module.embed_dim
                q_b, k_b, v_b = module.in_proj_bias.chunk(3, dim=0)
                new_module.q_proj.bias.copy_(q_b)
                new_module.k_proj.bias.copy_(k_b)
                new_module.v_proj.bias.copy_(v_b)

            # Output projection
            new_module.out_proj.weight.copy_(module.out_proj.weight)
            if module.out_proj.bias is not None and new_module.out_proj.bias is not None:
                new_module.out_proj.bias.copy_(module.out_proj.bias)

    return new_module


def replace_attention(
    module: nn.Module,
    variant: str = "pure_infsa",
    rho_init: float = 0.95,
    rho_trainable: bool = True,
    copy_weights: bool = True,
) -> InfSAAttention:
    """Replace a single attention module with InfSA.

    Handles ``nn.MultiheadAttention`` and attempts auto-detection for
    other attention module types.

    Args:
        module: The attention module to replace.
        variant: InfSA variant (``"pure_infsa"`` or ``"linear_infsa"``).
        rho_init: Initial ρ value.
        rho_trainable: Whether ρ is learnable.
        copy_weights: If True, copy Q/K/V/O projection weights from the
            source module.

    Returns:
        A new ``InfSAAttention`` module.

    Raises:
        ValueError: If the module's embed_dim/num_heads cannot be determined.
    """
    if isinstance(module, nn.MultiheadAttention):
        return _convert_mha(module, variant, rho_init, rho_trainable, copy_weights)

    # Generic fallback: try to extract dimensions
    embed_dim, num_heads = _get_embed_dim_and_heads(module)
    if embed_dim is None or num_heads is None:
        raise ValueError(
            f"Cannot auto-detect embed_dim/num_heads from {type(module).__name__}. "
            f"Got embed_dim={embed_dim}, num_heads={num_heads}. "
            f"Use InfSAAttention directly instead."
        )

    has_bias = any(
        hasattr(child, "bias") and child.bias is not None
        for _, child in module.named_children()
        if isinstance(child, nn.Linear)
    )
    dropout = getattr(module, "dropout", 0.0)
    if isinstance(dropout, nn.Module):
        dropout = getattr(dropout, "p", 0.0)
    batch_first = getattr(module, "batch_first", True)

    new_module = InfSAAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        variant=variant,
        dropout=dropout if isinstance(dropout, float) else 0.0,
        bias=has_bias,
        batch_first=batch_first,
        rho_init=rho_init,
        rho_trainable=rho_trainable,
    )

    if copy_weights:
        _try_copy_generic_weights(module, new_module)

    return new_module


def _try_copy_generic_weights(src: nn.Module, dst: InfSAAttention):
    """Best-effort weight copy from a generic attention module to InfSAAttention."""
    with torch.no_grad():
        # Try common attribute patterns for Q/K/V projections
        q_patterns = ["q_proj", "query", "q_linear", "Wq", "w_q", "q"]
        k_patterns = ["k_proj", "key", "k_linear", "Wk", "w_k", "k"]
        v_patterns = ["v_proj", "value", "v_linear", "Wv", "w_v", "v"]
        o_patterns = ["out_proj", "o_proj", "output", "dense", "out_linear", "c_proj"]

        def _find_and_copy(src_module, patterns, dst_linear):
            for pat in patterns:
                if hasattr(src_module, pat):
                    child = getattr(src_module, pat)
                    if isinstance(child, nn.Linear):
                        if child.weight.shape == dst_linear.weight.shape:
                            dst_linear.weight.copy_(child.weight)
                            if child.bias is not None and dst_linear.bias is not None:
                                dst_linear.bias.copy_(child.bias)
                            return True
            return False

        _find_and_copy(src, q_patterns, dst.q_proj)
        _find_and_copy(src, k_patterns, dst.k_proj)
        _find_and_copy(src, v_patterns, dst.v_proj)
        _find_and_copy(src, o_patterns, dst.out_proj)


def convert(
    model: nn.Module,
    variant: str = "pure_infsa",
    rho_init: float = 0.95,
    rho_trainable: bool = True,
    copy_weights: bool = True,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    target_types: Optional[List[Type]] = None,
    inplace: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """Replace all attention layers in a model with InfSA.

    Recursively walks the module tree, identifies attention layers, and
    replaces them with ``InfSAAttention``.

    Args:
        model: The model to convert.
        variant: InfSA variant (``"pure_infsa"`` or ``"linear_infsa"``).
        rho_init: Initial ρ value for all InfSA layers.
        rho_trainable: Whether ρ is learnable.
        copy_weights: If True, copy existing Q/K/V/O weights to InfSA modules.
        include_patterns: If specified, only replace modules whose full name
            matches one of these regex patterns. E.g., ``["encoder\\.layer\\..*"]``.
        exclude_patterns: If specified, skip modules whose full name matches
            any of these regex patterns.
        target_types: If specified, only replace modules of these types.
            Defaults to auto-detection.
        inplace: If True, modify the model in place. Otherwise, deepcopy first.
        verbose: If True, log which modules are replaced.

    Returns:
        The converted model (same object if inplace=True, copy otherwise).

    Examples::

        # Convert all attention in a torchvision ViT
        from torchvision.models import vit_b_16
        model = vit_b_16(weights="DEFAULT")
        model = infsa.convert(model, variant="pure_infsa")

        # Convert only specific layers
        model = infsa.convert(model, include_patterns=["encoder\\.layers\\.[0-5]\\."])

        # Convert a HuggingFace model
        from transformers import AutoModel
        model = AutoModel.from_pretrained("bert-base-uncased")
        model = infsa.convert(model, variant="linear_infsa")
    """
    if variant not in ("pure_infsa", "linear_infsa"):
        raise ValueError(
            f"Unknown variant '{variant}'. Choose 'pure_infsa' or 'linear_infsa'."
        )

    if not inplace:
        model = copy.deepcopy(model)

    replaced = 0
    skipped = 0

    # Collect (parent, attr_name, child) triples to replace
    replacements: List[Tuple[nn.Module, str, nn.Module, str]] = []

    for full_name, module in model.named_modules():
        # Check include/exclude patterns
        if include_patterns:
            if not any(re.search(pat, full_name) for pat in include_patterns):
                continue
        if exclude_patterns:
            if any(re.search(pat, full_name) for pat in exclude_patterns):
                continue

        # Check if this is an attention module
        is_target = False
        if target_types:
            is_target = isinstance(module, tuple(target_types))
        else:
            is_target = _is_attention_module(full_name, module)

        if not is_target:
            continue

        # Skip if already InfSAAttention
        if isinstance(module, InfSAAttention):
            continue

        replacements.append((full_name, module))

    for full_name, module in replacements:
        try:
            new_module = replace_attention(
                module,
                variant=variant,
                rho_init=rho_init,
                rho_trainable=rho_trainable,
                copy_weights=copy_weights,
            )

            # Navigate to parent and set attribute
            parts = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)

            attr_name = parts[-1]
            if attr_name.isdigit():
                parent[int(attr_name)] = new_module
            else:
                setattr(parent, attr_name, new_module)

            replaced += 1
            if verbose:
                logger.info(f"[InfSA] Replaced: {full_name} → InfSAAttention({variant})")

        except (ValueError, RuntimeError) as e:
            skipped += 1
            if verbose:
                logger.warning(f"[InfSA] Skipped: {full_name} — {e}")

    if verbose:
        print(
            f"[InfSA] Conversion complete: {replaced} attention layer(s) replaced"
            + (f", {skipped} skipped" if skipped else "")
            + f" (variant={variant}, rho_init={rho_init})"
        )

    if replaced == 0:
        print(
            "[InfSA] Warning: No attention layers were replaced. "
            "Ensure the model contains nn.MultiheadAttention or attention-like modules. "
            "You can also use target_types=[YourAttentionClass] to specify custom types."
        )

    return model
