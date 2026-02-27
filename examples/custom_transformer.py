"""
Example: Build a custom Transformer from scratch using InfSA.

Shows how to use the functional and module-level APIs to build
a Transformer encoder for any task (classification, detection, etc.)
that uses InfSA instead of softmax attention.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import infsa


class InfSATransformerEncoder(nn.Module):
    """Transformer Encoder using InfSA attention.

    Drop-in replacement for nn.TransformerEncoder that uses InfSA
    instead of softmax attention.

    Args:
        embed_dim: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of encoder layers.
        ff_dim: Feed-forward dimension. Default: 4 * embed_dim.
        variant: InfSA variant — "pure_infsa" or "linear_infsa".
        dropout: Dropout rate.
        rho_init: Initial ρ value.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = None,
        variant: str = "pure_infsa",
        dropout: float = 0.1,
        rho_init: float = 0.9,
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                InfSAEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    variant=variant,
                    dropout=dropout,
                    rho_init=rho_init,
                )
            )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class InfSAEncoderLayer(nn.Module):
    """Single Transformer encoder layer with InfSA attention."""

    def __init__(self, embed_dim, num_heads, ff_dim, variant, dropout, rho_init):
        super().__init__()
        self.attn = infsa.InfSAAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            variant=variant,
            dropout=dropout,
            batch_first=True,
            rho_init=rho_init,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm with residual
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x


class ImageClassifier(nn.Module):
    """Simple ViT-like image classifier using InfSA attention."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        variant: str = "pure_infsa",
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.encoder = InfSATransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            variant=variant,
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, E)
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        # Encode
        x = self.encoder(x)
        # Classify from CLS token
        return self.head(x[:, 0])


def main():
    print("=== Custom ViT with InfSA (pure_infsa) ===")
    model = ImageClassifier(
        img_size=224, patch_size=16, num_classes=100,
        embed_dim=384, num_heads=6, num_layers=6,
        variant="pure_infsa",
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output: {out.shape}")  # (2, 100)

    # Training sanity check
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    print("Training step OK!")

    # Show rho values per layer
    print("\nLearnable ρ values:")
    for name, param in model.named_parameters():
        if "rho" in name:
            rho = torch.sigmoid(param).item()
            print(f"  {name}: ρ = {rho:.4f}")

    print("\n=== Custom ViT with InfSA (linear_infsa) ===")
    model2 = ImageClassifier(
        img_size=224, patch_size=16, num_classes=100,
        embed_dim=384, num_heads=6, num_layers=6,
        variant="linear_infsa",
    )
    x = torch.randn(2, 3, 224, 224)
    out2 = model2(x)
    print(f"Output: {out2.shape}")
    print("linear_infsa variant works correctly!")


if __name__ == "__main__":
    main()
