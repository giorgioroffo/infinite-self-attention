"""
Integration test: HuggingFace ViT (google/vit-base-patch16-224) with InfSA.

Loads a pretrained ViT from HuggingFace, replaces its attention mechanism with
InfSA using a compatible wrapper, runs inference on synthetic ImageNet-shaped
inputs (using annotation metadata from datasets/imnet/), and validates outputs
+ gradient flow.

Results are written to results/test_huggingface_vit.txt
"""

import sys
import os
from datetime import datetime

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import infsa
from helpers import get_device, load_imnet_annotations, write_results


class HFViTInfSAAttention(nn.Module):
    """Wrapper that adapts InfSAAttention to HuggingFace ViTSelfAttention interface.

    HF ViT calls attention with (hidden_states, head_mask, output_attentions).
    This wrapper translates to InfSAAttention's (query, key, value) interface.
    """
    def __init__(self, embed_dim, num_heads, variant="pure_infsa", rho_init=0.9):
        super().__init__()
        self.infsa_attn = infsa.InfSAAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            variant=variant, batch_first=True, rho_init=rho_init,
        )
        self.num_attention_heads = num_heads
        self.attention_head_size = embed_dim // num_heads
        self.all_head_size = embed_dim

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        attn_out, attn_weights = self.infsa_attn(
            hidden_states, hidden_states, hidden_states,
            need_weights=output_attentions,
        )
        return (attn_out, attn_weights)


def replace_hf_vit_attention(model, variant, rho_init=0.9):
    """Replace ViTSelfAttention modules inside a HF ViT with InfSA wrappers."""
    from transformers.models.vit.modeling_vit import ViTSelfAttention
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, ViTSelfAttention):
            embed_dim = module.all_head_size
            num_heads = module.num_attention_heads
            wrapper = HFViTInfSAAttention(embed_dim, num_heads, variant, rho_init)
            # Copy Q/K/V weights
            with torch.no_grad():
                wrapper.infsa_attn.q_proj.weight.copy_(module.query.weight)
                wrapper.infsa_attn.q_proj.bias.copy_(module.query.bias)
                wrapper.infsa_attn.k_proj.weight.copy_(module.key.weight)
                wrapper.infsa_attn.k_proj.bias.copy_(module.key.bias)
                wrapper.infsa_attn.v_proj.weight.copy_(module.value.weight)
                wrapper.infsa_attn.v_proj.bias.copy_(module.value.bias)
            # Navigate to parent and set
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], wrapper)
            else:
                setattr(model, name, wrapper)
            replaced += 1
    return model, replaced


def _get_device():
    return get_device()


def run_test():
    results = []
    results.append(f"Test: HuggingFace ViT + InfSA")
    results.append(f"Date: {datetime.now().isoformat()}")
    results.append(f"PyTorch: {torch.__version__}")
    results.append("=" * 60)

    device = _get_device()
    results.append(f"Device: {device}")

    all_passed = True

    # --- Load annotations ---
    try:
        annotations = load_imnet_annotations()
        results.append(f"Loaded {len(annotations)} ImageNet annotations")
        assert len(annotations) > 0, "No annotations found"
        results.append("  [PASSED] Dataset loading")
    except Exception as e:
        results.append(f"  [FAILED] Dataset loading: {e}")
        all_passed = False
        annotations = [{"filename": "dummy", "width": 224, "height": 224, "class_id": "n01440764"}]

    # --- Test both variants ---
    for variant in ["pure_infsa", "linear_infsa"]:
        results.append(f"\n--- Variant: {variant} ---")

        # 1. Load model
        try:
            from transformers import ViTForImageClassification

            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
            model.eval()
            results.append(f"  [PASSED] Model loading (ViTForImageClassification)")
        except Exception as e:
            results.append(f"  [FAILED] Model loading: {e}")
            all_passed = False
            continue

        # 2. Convert to InfSA
        try:
            model, infsa_count = replace_hf_vit_attention(model, variant=variant, rho_init=0.9)
            assert infsa_count > 0, "No InfSA layers found after conversion"
            results.append(f"  [PASSED] Conversion ({infsa_count} InfSA layers)")
        except Exception as e:
            results.append(f"  [FAILED] Conversion: {e}")
            all_passed = False
            continue

        model = model.to(device)

        # 3. Inference on dataset samples
        try:
            model.eval()
            num_samples = min(5, len(annotations))
            for i, ann in enumerate(annotations[:num_samples]):
                dummy_input = torch.randn(1, 3, 224, 224, device=device)
                with torch.no_grad():
                    output = model(dummy_input)
                    logits = output.logits
                    assert logits.shape == (1, 1000), f"Unexpected shape: {logits.shape}"
                    pred_class = logits.argmax(dim=-1).item()
                results.append(f"    Sample {i+1} ({ann['filename']}): pred_class={pred_class}, shape={logits.shape}")
            results.append(f"  [PASSED] Inference ({num_samples} samples)")
        except Exception as e:
            results.append(f"  [FAILED] Inference: {e}")
            all_passed = False
            continue

        # 4. Gradient flow
        try:
            model.train()
            x = torch.randn(2, 3, 224, 224, device=device)
            output = model(x)
            loss = output.logits.sum()
            loss.backward()
            params_with_grad = sum(1 for _, p in model.named_parameters()
                                  if p.grad is not None and p.grad.abs().sum() > 0)
            assert params_with_grad > 0, "No parameters received gradients"
            results.append(f"  [PASSED] Gradient flow ({params_with_grad} params with gradients)")
        except Exception as e:
            results.append(f"  [FAILED] Gradient flow: {e}")
            all_passed = False

        # 5. Training step
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            x = torch.randn(2, 3, 224, 224, device=device)
            output = model(x)
            loss = output.logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            results.append(f"  [PASSED] Training step (loss={loss.item():.4f})")
        except Exception as e:
            results.append(f"  [FAILED] Training step: {e}")
            all_passed = False

    # --- Summary ---
    results.append("\n" + "=" * 60)
    status = "PASSED" if all_passed else "FAILED"
    results.append(f"OVERALL: {status}")

    write_results(results, "test_huggingface_vit.txt")
    return all_passed


if __name__ == "__main__":
    passed = run_test()
    sys.exit(0 if passed else 1)
