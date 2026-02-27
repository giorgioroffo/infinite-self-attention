"""
Example: Replace attention in a torchvision ViT with InfSA.

This example shows how to convert a pretrained ViT-B/16 from torchvision
to use InfSA attention, preserving all pretrained weights.
"""

import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import infsa


def main():
    # 1. Load pretrained ViT
    print("Loading pretrained ViT-B/16...")
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.eval()

    # 2. Convert all attention to InfSA (pure_infsa variant)
    #    Weights from pretrained Q/K/V projections are automatically copied.
    model = infsa.convert(model, variant="pure_infsa", rho_init=0.9)

    # 3. Run inference
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # (1, 1000)

    # 4. Fine-tune (ρ is trainable by default)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for step in range(3):
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step {step+1}, loss: {loss.item():.4f}")

    print("\nDone! ViT-B/16 is now using InfSA attention.")


if __name__ == "__main__":
    main()
