"""
Example: Replace attention in a HuggingFace Vision Transformer with InfSA.

Shows how to convert a pretrained HuggingFace ViT (e.g., google/vit-base-patch16-224)
to use InfSA attention.
"""

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import infsa


def main():
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
    except ImportError:
        print("Install transformers: pip install transformers")
        return

    # 1. Load pretrained ViT from HuggingFace
    print("Loading HuggingFace ViT-Base...")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.eval()

    # 2. Inspect attention modules before conversion
    print("\nBefore conversion:")
    for name, module in model.named_modules():
        cls = type(module).__name__
        if "attention" in cls.lower() or "attention" in name.lower():
            print(f"  {name}: {cls}")

    # 3. Convert attention to InfSA
    #    For HuggingFace models, use target_types to specify which modules to replace.
    #    The ViTSelfAttention class is the standard HF ViT attention.
    from transformers.models.vit.modeling_vit import ViTSelfAttention
    model = infsa.convert(
        model,
        variant="pure_infsa",
        target_types=[ViTSelfAttention],
        rho_init=0.9,
    )

    # 4. Verify conversion
    print("\nAfter conversion:")
    infsa_count = 0
    for name, module in model.named_modules():
        if isinstance(module, infsa.InfSAAttention):
            infsa_count += 1
            print(f"  {name}: InfSAAttention(rho={module.rho:.3f})")
    print(f"\nTotal InfSA layers: {infsa_count}")

    # 5. Run inference
    dummy_input = torch.randn(1, 3, 224, 224)
    # Note: HF ViT expects pixel_values, but we can create a dummy
    print(f"\nModel converted successfully. Ready for fine-tuning!")


if __name__ == "__main__":
    main()
