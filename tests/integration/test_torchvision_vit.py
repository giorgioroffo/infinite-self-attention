"""
Integration test: torchvision ViT-B/16 with InfSA.

Loads a pretrained ViT-B/16 from torchvision, converts its attention layers to
InfSA, runs inference on synthetic ImageNet-shaped inputs (using annotation
metadata from datasets/imnet/), and validates outputs + gradient flow.

Results are written to results/test_torchvision_vit.txt
"""

import sys
import os
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import infsa
from helpers import get_device, load_imnet_annotations, write_results


def run_test():
    results = []
    results.append(f"Test: torchvision ViT-B/16 + InfSA")
    results.append(f"Date: {datetime.now().isoformat()}")
    results.append(f"PyTorch: {torch.__version__}")
    results.append("=" * 60)

    device = get_device()
    results.append(f"Device: {device}")

    all_passed = True

    # --- Load annotations ---
    try:
        annotations = load_imnet_annotations()
        results.append(f"Loaded {len(annotations)} ImageNet annotations")
        assert len(annotations) > 0
        results.append("  [PASSED] Dataset loading")
    except Exception as e:
        results.append(f"  [FAILED] Dataset loading: {e}")
        all_passed = False
        annotations = [{"filename": "dummy", "width": 224, "height": 224,
                        "class_id": "n01440764", "bbox": None}]

    # --- Test both variants ---
    for variant in ["pure_infsa", "linear_infsa"]:
        results.append(f"\n--- Variant: {variant} ---")

        # 1. Load model
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            model.eval()
            results.append(f"  [PASSED] Model loading (ViT-B/16 pretrained)")
        except Exception as e:
            results.append(f"  [FAILED] Model loading: {e}")
            all_passed = False
            continue

        # 2. Convert to InfSA
        try:
            model = infsa.convert(
                model,
                variant=variant,
                rho_init=0.9,
                verbose=False,
            )
            infsa_count = sum(1 for _, m in model.named_modules() if isinstance(m, infsa.InfSAAttention))
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
                    assert output.shape == (1, 1000), f"Unexpected shape: {output.shape}"
                    pred_class = output.argmax(dim=-1).item()
                results.append(f"    Sample {i+1} ({ann['filename']}): pred_class={pred_class}, shape={output.shape}")
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
            loss = output.sum()
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
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            results.append(f"  [PASSED] Training step (loss={loss.item():.4f})")
        except Exception as e:
            results.append(f"  [FAILED] Training step: {e}")
            all_passed = False

        # 6. Verify rho values
        try:
            rho_vals = []
            for name, param in model.named_parameters():
                if "rho" in name:
                    rho = torch.sigmoid(param).item()
                    rho_vals.append(rho)
            assert len(rho_vals) > 0
            results.append(f"  [PASSED] Rho values: min={min(rho_vals):.4f}, max={max(rho_vals):.4f}")
        except Exception as e:
            results.append(f"  [FAILED] Rho values: {e}")
            all_passed = False

    # --- Summary ---
    results.append("\n" + "=" * 60)
    status = "PASSED" if all_passed else "FAILED"
    results.append(f"OVERALL: {status}")

    write_results(results, "test_torchvision_vit.txt")
    return all_passed


if __name__ == "__main__":
    passed = run_test()
    sys.exit(0 if passed else 1)
