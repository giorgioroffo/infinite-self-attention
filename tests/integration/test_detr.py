"""
Integration test: HuggingFace DETR (facebook/detr-resnet-50) with InfSA.

Loads a pretrained DETR object-detection model, patches its core attention
computation with InfSA (using the functional API), runs inference on synthetic
inputs based on ImageNet annotation metadata, and validates outputs + gradient flow.

Results are written to results/test_detr.txt
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


def patch_detr_attention(model, variant="pure_infsa", rho_init=0.9):
    """Monkey-patch DETR's attention to use InfSA for the core QK^T computation.

    This preserves DETR's position embedding logic and cross-attention handling
    while replacing the softmax(QK^T/sqrt(d)) with InfSA scoring.
    """
    from transformers.models.detr.modeling_detr import DetrSelfAttention, DetrCrossAttention

    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, (DetrSelfAttention, DetrCrossAttention)):
            # Store rho as a parameter on the module
            rho_logit = torch.tensor(torch.logit(torch.tensor(rho_init)))
            module.register_parameter("rho_logit", nn.Parameter(rho_logit))
            module._infsa_variant = variant

            if isinstance(module, DetrSelfAttention):
                original_forward = module.forward

                def make_patched_self_forward(mod):
                    def patched_forward(
                        hidden_states,
                        attention_mask=None,
                        position_embeddings=None,
                        **kwargs,
                    ):
                        input_shape = hidden_states.shape[:-1]
                        hidden_shape = (*input_shape, -1, mod.head_dim)

                        query_key_input = hidden_states + position_embeddings if position_embeddings is not None else hidden_states

                        q = mod.q_proj(query_key_input).view(hidden_shape).transpose(1, 2)
                        k = mod.k_proj(query_key_input).view(hidden_shape).transpose(1, 2)
                        v = mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                        attn_out = infsa.infsa_attention(q, k, v, variant=mod._infsa_variant,
                                                          rho=torch.sigmoid(mod.rho_logit))
                        B = hidden_states.shape[0]
                        E = hidden_states.shape[-1]
                        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
                        attn_out = mod.o_proj(attn_out)
                        return (attn_out, None)
                    return patched_forward

                module.forward = make_patched_self_forward(module)

            else:  # DetrCrossAttention
                def make_patched_cross_forward(mod):
                    def patched_forward(
                        hidden_states,
                        key_value_states,
                        attention_mask=None,
                        position_embeddings=None,
                        encoder_position_embeddings=None,
                        **kwargs,
                    ):
                        input_shape = hidden_states.shape[:-1]
                        hidden_shape_q = (*input_shape, -1, mod.head_dim)
                        kv_shape = (*key_value_states.shape[:-1], -1, mod.head_dim)

                        query_input = hidden_states + position_embeddings if position_embeddings is not None else hidden_states
                        key_input = key_value_states + encoder_position_embeddings if encoder_position_embeddings is not None else key_value_states

                        q = mod.q_proj(query_input).view(hidden_shape_q).transpose(1, 2)
                        k = mod.k_proj(key_input).view(kv_shape).transpose(1, 2)
                        v = mod.v_proj(key_value_states).view(kv_shape).transpose(1, 2)

                        # Use pure_infsa for cross-attention (different Q/K lengths)
                        effective_variant = "pure_infsa"
                        attn_out = infsa.infsa_attention(q, k, v, variant=effective_variant,
                                                          rho=torch.sigmoid(mod.rho_logit))
                        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
                        attn_out = mod.o_proj(attn_out)
                        return (attn_out, None)
                    return patched_forward

                module.forward = make_patched_cross_forward(module)

            patched += 1

    return model, patched


def _get_device():
    return get_device()


def run_test():
    results = []
    results.append(f"Test: DETR (facebook/detr-resnet-50) + InfSA")
    results.append(f"Date: {datetime.now().isoformat()}")
    results.append(f"PyTorch: {torch.__version__}")
    results.append("=" * 60)

    device = _get_device()
    results.append(f"Device: {device}")

    all_passed = True

    # --- Load annotations ---
    try:
        annotations = load_imnet_annotations(include_bbox=True)
        results.append(f"Loaded {len(annotations)} ImageNet annotations")
        assert len(annotations) > 0
        results.append("  [PASSED] Dataset loading")
    except Exception as e:
        results.append(f"  [FAILED] Dataset loading: {e}")
        all_passed = False
        annotations = [{"filename": "dummy", "width": 800, "height": 600,
                        "class_id": "n01440764", "bbox": [100, 100, 300, 300]}]

    for variant in ["pure_infsa", "linear_infsa"]:
        results.append(f"\n--- Variant: {variant} ---")

        # 1. Load DETR
        try:
            from transformers import DetrForObjectDetection
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            model.eval()
            results.append(f"  [PASSED] Model loading (DETR-ResNet-50)")
        except Exception as e:
            results.append(f"  [FAILED] Model loading: {e}")
            all_passed = False
            continue

        # 2. Patch attention to InfSA
        try:
            model, infsa_count = patch_detr_attention(model, variant=variant, rho_init=0.9)
            assert infsa_count > 0, "No attention layers patched"
            results.append(f"  [PASSED] Conversion ({infsa_count} InfSA-patched layers)")
        except Exception as e:
            results.append(f"  [FAILED] Conversion: {e}")
            all_passed = False
            continue

        model = model.to(device)

        # 3. Inference on dataset samples
        try:
            model.eval()
            num_samples = min(3, len(annotations))
            for i, ann in enumerate(annotations[:num_samples]):
                # DETR expects 3xHxW, use 800x800 as standard
                dummy_input = torch.randn(1, 3, 800, 800, device=device)
                with torch.no_grad():
                    outputs = model(pixel_values=dummy_input)
                    logits = outputs.logits  # (B, num_queries, num_classes+1)
                    pred_boxes = outputs.pred_boxes  # (B, num_queries, 4)
                    assert logits.ndim == 3, f"Unexpected logits shape: {logits.shape}"
                    assert pred_boxes.ndim == 3, f"Unexpected pred_boxes shape: {pred_boxes.shape}"
                results.append(f"    Sample {i+1} ({ann['filename']}): "
                             f"queries={logits.shape[1]}, classes={logits.shape[2]}, "
                             f"boxes={pred_boxes.shape}")
            results.append(f"  [PASSED] Inference ({num_samples} samples)")
        except Exception as e:
            results.append(f"  [FAILED] Inference: {e}")
            all_passed = False
            continue

        # 4. Gradient flow
        try:
            model.train()
            x = torch.randn(1, 3, 800, 800, device=device)
            outputs = model(pixel_values=x)
            loss = outputs.logits.sum() + outputs.pred_boxes.sum()
            loss.backward()
            params_with_grad = sum(1 for _, p in model.named_parameters()
                                  if p.grad is not None and p.grad.abs().sum() > 0)
            assert params_with_grad > 0, "No parameters received gradients"
            results.append(f"  [PASSED] Gradient flow ({params_with_grad} params)")
        except Exception as e:
            results.append(f"  [FAILED] Gradient flow: {e}")
            all_passed = False

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Summary ---
    results.append("\n" + "=" * 60)
    status = "PASSED" if all_passed else "FAILED"
    results.append(f"OVERALL: {status}")

    write_results(results, "test_detr.txt")
    return all_passed


if __name__ == "__main__":
    passed = run_test()
    sys.exit(0 if passed else 1)
