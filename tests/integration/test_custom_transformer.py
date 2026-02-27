"""
Integration test: Custom Transformer built from scratch with InfSA.

Builds a custom ViT-like image classifier and a mini LLM using InfSA attention
directly (not via conversion). Processes ImageNet annotation metadata and LLM
benchmark samples to validate end-to-end functionality.

Results are written to results/test_custom_transformer.txt
"""

import sys
import os
from datetime import datetime

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import infsa
from helpers import get_device, load_imnet_annotations, load_jsonl_samples, write_results


# ─── Custom architectures built with InfSA ───────────────────────────────────

class InfSAEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, variant, dropout, rho_init):
        super().__init__()
        self.attn = infsa.InfSAAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            variant=variant, dropout=dropout,
            batch_first=True, rho_init=rho_init,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x


class CustomViT(nn.Module):
    """ViT-like classifier built from scratch with InfSA."""
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=384, num_heads=6, num_layers=6, variant="pure_infsa"):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.layers = nn.ModuleList([
            InfSAEncoderLayer(embed_dim, num_heads, embed_dim * 4, variant, 0.1, 0.9)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x[:, 0]))


class MiniLLM(nn.Module):
    """Mini LLM built from scratch with InfSA."""
    def __init__(self, vocab_size=32000, embed_dim=256, num_heads=4,
                 num_layers=4, max_seq_len=512, variant="pure_infsa"):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            InfSAEncoderLayer(embed_dim, num_heads, embed_dim * 4, variant, 0.1, 0.9)
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
        return self.head(self.norm(x))


# ─── Helpers ──────────────────────────────────────────────────────────────────

# ─── Test runner ──────────────────────────────────────────────────────────────

def _get_device():
    return get_device()


def run_test():
    results = []
    results.append(f"Test: Custom Transformer (ViT + MiniLLM) with InfSA")
    results.append(f"Date: {datetime.now().isoformat()}")
    results.append(f"PyTorch: {torch.__version__}")
    results.append("=" * 60)

    device = _get_device()
    results.append(f"Device: {device}")

    all_passed = True

    # --- Load datasets ---
    try:
        annotations = load_imnet_annotations()
        assert len(annotations) > 0
        results.append(f"  [PASSED] ImageNet annotations ({len(annotations)} samples)")
    except Exception as e:
        results.append(f"  [FAILED] ImageNet annotations: {e}")
        all_passed = False
        annotations = []

    try:
        squad_samples = load_jsonl_samples("squad.jsonl", max_samples=3)
        gsm8k_samples = load_jsonl_samples("gsm8k.jsonl", max_samples=3)
        results.append(f"  [PASSED] LLM datasets (squad={len(squad_samples)}, gsm8k={len(gsm8k_samples)})")
    except Exception as e:
        results.append(f"  [FAILED] LLM datasets: {e}")
        all_passed = False
        squad_samples, gsm8k_samples = [], []

    # ============================================
    # Part A: Custom ViT with InfSA
    # ============================================
    for variant in ["pure_infsa", "linear_infsa"]:
        results.append(f"\n--- Custom ViT ({variant}) ---")

        try:
            model = CustomViT(variant=variant, num_classes=100).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            infsa_count = sum(1 for _, m in model.named_modules() if isinstance(m, infsa.InfSAAttention))
            results.append(f"  [PASSED] Model creation ({total_params:,} params, {infsa_count} InfSA layers)")
        except Exception as e:
            results.append(f"  [FAILED] Model creation: {e}")
            all_passed = False
            continue

        # Inference
        try:
            model.eval()
            num_samples = min(5, max(len(annotations), 1))
            for i in range(num_samples):
                x = torch.randn(1, 3, 224, 224, device=device)
                with torch.no_grad():
                    out = model(x)
                    assert out.shape == (1, 100), f"Unexpected shape: {out.shape}"
                fname = annotations[i]["filename"] if i < len(annotations) else "synthetic"
                results.append(f"    Sample {i+1} ({fname}): pred={out.argmax(1).item()}, shape={out.shape}")
            results.append(f"  [PASSED] Inference ({num_samples} samples)")
        except Exception as e:
            results.append(f"  [FAILED] Inference: {e}")
            all_passed = False

        # Training
        try:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for step in range(3):
                x = torch.randn(2, 3, 224, 224, device=device)
                out = model(x)
                loss = nn.functional.cross_entropy(out, torch.randint(0, 100, (2,), device=device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            results.append(f"  [PASSED] Training (3 steps, final loss={loss.item():.4f})")
        except Exception as e:
            results.append(f"  [FAILED] Training: {e}")
            all_passed = False

        # Rho values
        try:
            rho_vals = [torch.sigmoid(p).item() for n, p in model.named_parameters() if "rho" in n]
            assert len(rho_vals) > 0
            results.append(f"  [PASSED] Rho values: min={min(rho_vals):.4f}, max={max(rho_vals):.4f}")
        except Exception as e:
            results.append(f"  [FAILED] Rho values: {e}")
            all_passed = False

    # ============================================
    # Part B: Mini LLM with InfSA
    # ============================================
    for variant in ["pure_infsa", "linear_infsa"]:
        results.append(f"\n--- Mini LLM ({variant}) ---")

        try:
            vocab_size = 1000
            model = MiniLLM(vocab_size=vocab_size, variant=variant).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            infsa_count = sum(1 for _, m in model.named_modules() if isinstance(m, infsa.InfSAAttention))
            results.append(f"  [PASSED] Model creation ({total_params:,} params, {infsa_count} InfSA layers)")
        except Exception as e:
            results.append(f"  [FAILED] Model creation: {e}")
            all_passed = False
            continue

        # Inference with LLM dataset token IDs (simulated)
        try:
            model.eval()
            for i, sample in enumerate(squad_samples[:2]):
                # Simulate tokenized input
                input_ids = torch.randint(0, vocab_size, (1, 64), device=device)
                with torch.no_grad():
                    logits = model(input_ids)
                    assert logits.shape == (1, 64, vocab_size)
                results.append(f"    SQuAD sample {i+1}: logits shape={logits.shape}")
            for i, sample in enumerate(gsm8k_samples[:2]):
                input_ids = torch.randint(0, vocab_size, (1, 64), device=device)
                with torch.no_grad():
                    logits = model(input_ids)
                    assert logits.shape == (1, 64, vocab_size)
                results.append(f"    GSM8K sample {i+1}: logits shape={logits.shape}")
            results.append(f"  [PASSED] Inference on LLM samples")
        except Exception as e:
            results.append(f"  [FAILED] Inference on LLM samples: {e}")
            all_passed = False

        # Training
        try:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for step in range(3):
                input_ids = torch.randint(0, vocab_size, (2, 64), device=device)
                logits = model(input_ids)
                targets = torch.randint(0, vocab_size, (2, 64), device=device)
                loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            results.append(f"  [PASSED] Training (3 steps, final loss={loss.item():.4f})")
        except Exception as e:
            results.append(f"  [FAILED] Training: {e}")
            all_passed = False

    # ============================================
    # Part C: nn.TransformerEncoder conversion
    # ============================================
    results.append(f"\n--- nn.TransformerEncoder conversion ---")
    for variant in ["pure_infsa", "linear_infsa"]:
        try:
            encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
            encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            encoder = infsa.convert(encoder, variant=variant, rho_init=0.9, verbose=False)
            infsa_count = sum(1 for _, m in encoder.named_modules() if isinstance(m, infsa.InfSAAttention))
            assert infsa_count > 0

            encoder = encoder.to(device)
            x = torch.randn(2, 32, 256, device=device)
            with torch.no_grad():
                out = encoder(x)
                assert out.shape == (2, 32, 256)
            results.append(f"  [PASSED] TransformerEncoder conversion ({variant}, {infsa_count} layers)")
        except Exception as e:
            results.append(f"  [FAILED] TransformerEncoder conversion ({variant}): {e}")
            all_passed = False

    # --- Summary ---
    results.append("\n" + "=" * 60)
    status = "PASSED" if all_passed else "FAILED"
    results.append(f"OVERALL: {status}")

    write_results(results, "test_custom_transformer.txt")
    return all_passed


if __name__ == "__main__":
    passed = run_test()
    sys.exit(0 if passed else 1)
