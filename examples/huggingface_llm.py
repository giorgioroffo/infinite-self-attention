"""
Example: Replace attention in a HuggingFace LLM with InfSA.

Shows multiple approaches for integrating InfSA into language models:

1. **Functional API** — Directly call infsa.infsa_attention() in a custom forward.
2. **Module replacement** — Use infsa.convert() to auto-replace attention modules.
3. **Manual injection** — Create InfSAAttention and wire it into the model.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import infsa


# ========================
# Approach 1: Functional API
# ========================

def custom_attention_forward(q, k, v, variant="pure_infsa", rho=0.9):
    """
    Use InfSA as a drop-in replacement in any custom attention implementation.

    This is the most flexible approach — works with any architecture where
    you have access to the Q, K, V tensors after projection.

    Args:
        q, k, v: Pre-projected tensors of shape (B, H, N, D).
    """
    return infsa.infsa_attention(q, k, v, variant=variant, rho=rho)


# ========================
# Approach 2: Auto-convert HuggingFace model
# ========================

def convert_huggingface_llm():
    """
    Automatically replace attention in a HuggingFace LLM.

    Works with GPT-2, BERT, LLaMA, Mistral, and other models that use
    standard attention module patterns.
    """
    try:
        from transformers import AutoModel
    except ImportError:
        print("Install transformers: pip install transformers")
        return

    print("Loading GPT-2...")
    model = AutoModel.from_pretrained("gpt2")

    # GPT-2 uses GPT2Attention. Auto-detect and replace.
    model = infsa.convert(model, variant="pure_infsa", rho_init=0.9)

    # Verify
    for name, module in model.named_modules():
        if isinstance(module, infsa.InfSAAttention):
            print(f"  Converted: {name}")

    return model


# ========================
# Approach 3: Manual injection into custom Transformer
# ========================

class CustomTransformerBlock(nn.Module):
    """
    Example of manually using InfSAAttention in a custom Transformer block.
    This approach gives you full control.
    """

    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, variant="pure_infsa"):
        super().__init__()

        # Use InfSA instead of nn.MultiheadAttention
        self.attn = infsa.InfSAAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            variant=variant,
            batch_first=True,
            rho_init=0.9,
            rho_trainable=True,  # ρ adapts during training
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        # Pre-norm Transformer block
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x


class MiniLLM(nn.Module):
    """A minimal LLM-like model using InfSA attention."""

    def __init__(self, vocab_size=32000, embed_dim=512, num_heads=8,
                 num_layers=6, max_seq_len=1024, variant="pure_infsa"):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            CustomTransformerBlock(embed_dim, num_heads, variant=variant)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


def demo_mini_llm():
    """Demonstrate InfSA in a small language model."""
    print("\n=== Mini LLM with InfSA ===")
    model = MiniLLM(
        vocab_size=1000, embed_dim=256, num_heads=4,
        num_layers=4, variant="pure_infsa"
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    rho_params = sum(p.numel() for n, p in model.named_parameters() if "rho" in n)
    print(f"Total params: {total_params:,}")
    print(f"Trainable rho params: {rho_params}")

    # Training step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 1000, (2, 64))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()
    optimizer.step()
    print(f"Output shape: {logits.shape}")  # (2, 64, 1000)
    print("Training step completed successfully!")


def main():
    # Approach 1: Functional API demo
    print("=== Approach 1: Functional API ===")
    q = torch.randn(2, 8, 64, 64)  # (B, H, N, D)
    k = torch.randn(2, 8, 64, 64)
    v = torch.randn(2, 8, 64, 64)
    out = custom_attention_forward(q, k, v, variant="pure_infsa")
    print(f"Functional output shape: {out.shape}")

    # Approach 2: Auto-conversion
    print("\n=== Approach 2: Auto-convert HuggingFace ===")
    try:
        convert_huggingface_llm()
    except Exception as e:
        print(f"Skipped (transformers not installed or model unavailable): {e}")

    # Approach 3: Manual injection
    demo_mini_llm()


if __name__ == "__main__":
    main()
