# Infinite Self-Attention (InfSA) / Linear-InfSA

Official implementation of the paper:
**Self-Attention And Beyond the Infinite: Towards Linear Transformers with Infinite Self-Attention**

Authors: Giorgio Roffo, Luke Palmer

Paper: arXiv:2603.00175

<p align="center">
  <img src="figures/gr_figure_method.png" width="85%"/>
</p>

<h1 align="center">InfSA — Infinite Self-Attention</h1>

<p align="center">
  <b>A spectral, graph-theoretic reformulation of self-attention for Transformers</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.00175">Paper (arXiv)</a> •
  <a href="#installation">Install</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#use-with-llms">LLMs</a> •
  <a href="#use-with-vision-models">Vision Models</a> •
  <a href="#api-reference">API</a> •
  <a href="#experimental-results">Results</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%3E%3D1.13-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Python-%3E%3D3.8-blue?logo=python" alt="Python">
</p>

---

**InfSA** (Infinite Self-Attention) replaces standard `softmax(QK^T/√d)` attention with a spectral diffusion mechanism grounded in graph centrality theory. Each attention layer becomes a diffusion step on a content-adaptive token graph, and token importance is determined by **eigenvector centrality** — the same principle behind PageRank and Katz ranking — rather than by local query-key affinity.

<p align="center">
  <img src="figures/gr_attention_graph.png" width="85%"/>
  <br>
  <em>Softmax attention distributes focus across background regions, while InfSA variants produce sharper, object-aligned activations.</em>
</p>

### Key properties

- **Drop-in compatible** — works with any Transformer: ViT, DINO, RT-DETR, GPT, LLaMA, Qwen, BERT, etc.
- **Two variants**: `pure_infsa` (quadratic, highest quality) and `linear_infsa` (O(N), scales to 332K tokens)
- **Learnable ρ** — the spectral decay parameter adapts during training via sigmoid reparameterization
- **One-liner conversion** — `model = infsa.convert(model, variant="pure_infsa")`
- **13× faster** than standard ViT at 1024² resolution (Linear-InfSA)
- **+3.2 pp** ImageNet-1K accuracy gain (84.7% vs 81.5%) as a pure architectural improvement

---

## Installation

```bash
pip install infsa
```

Or from source:

```bash
cd InfSA_release
pip install -e .
```

**Requirements**: `torch >= 1.13.0`. No other dependencies.

---

## Quick Start

### One-liner: convert any existing model

```python
import infsa

# Replace ALL attention layers with InfSA — weights are copied automatically
model = infsa.convert(model, variant="pure_infsa")
```

That's it. Works with torchvision, HuggingFace, timm, or any custom Transformer.

---

## Use with Vision Models

### Torchvision ViT

```python
import infsa
from torchvision.models import vit_b_16, ViT_B_16_Weights

model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
model = infsa.convert(model, variant="pure_infsa")
# All 12 attention layers now use InfSA. Fine-tune as usual.
```

### DINO / DINOv2

```python
import torch
import infsa

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model = infsa.convert(model, variant="pure_infsa", rho_init=0.9)
# DINOv2 ViT-B/14 now uses InfSA attention in all 12 layers
```

### HuggingFace ViT

```python
import infsa
from transformers import ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTSelfAttention

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model = infsa.convert(model, variant="pure_infsa", target_types=[ViTSelfAttention])
```

### RT-DETR (Object Detection)

```python
import infsa
from transformers import RTDetrForObjectDetection

model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
# Convert encoder attention to InfSA
model = infsa.convert(
    model, variant="pure_infsa",
    include_patterns=[r"encoder\.layers\."],
)
```

### timm ViT / Swin / DeiT

```python
import timm
import infsa

model = timm.create_model('vit_base_patch16_224', pretrained=True)
model = infsa.convert(model, variant="linear_infsa")
```

---

## Use with LLMs

InfSA is **modality-agnostic** — the graph diffusion principles apply to language tokens just as they do to image patches.

### Qwen3-8B

```python
import infsa
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto")
# Convert all attention layers to InfSA
model = infsa.convert(model, variant="pure_infsa", rho_init=0.9)
# Fine-tune with your data — ρ is learnable per layer
```

### GPT-2

```python
import infsa
from transformers import AutoModel

model = AutoModel.from_pretrained("gpt2")
model = infsa.convert(model, variant="pure_infsa")
# All 12 attention layers replaced ✓
```

### LLaMA / Mistral / Gemma

```python
import infsa
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = infsa.convert(model, variant="linear_infsa", rho_init=0.9)
```

### Custom LLM from scratch

```python
import infsa
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=4096, num_heads=32):
        super().__init__()
        self.attn = infsa.InfSAAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            variant="pure_infsa",
            rho_init=0.9,
            rho_trainable=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x
```

### Functional API (maximum flexibility)

For custom attention implementations where you control Q/K/V directly:

```python
import infsa

# Inside your custom forward method, after Q/K/V projection:
# q, k, v shape: (batch, heads, seq_len, head_dim)
output = infsa.infsa_attention(q, k, v, variant="pure_infsa", rho=0.9)
```

---

## Selective Conversion

Convert only specific layers, or exclude certain parts of a model:

```python
# Only convert the first 6 layers
model = infsa.convert(model, include_patterns=[r"layer\.[0-5]\."])

# Convert everything except the decoder
model = infsa.convert(model, exclude_patterns=[r"decoder\."])

# Only convert specific module types
from some_model import CustomAttention
model = infsa.convert(model, target_types=[CustomAttention])
```

---

## InfSA Variants

| Variant | Complexity | Output | Best for |
|---------|-----------|--------|----------|
| `pure_infsa` | O(N²·D) | N×N attention matrix | Quality-critical tasks, short-medium sequences |
| `linear_infsa` | O(N·D) | N×1 importance vector | Long sequences, high resolution, efficiency-critical |

### How it works

**Standard softmax attention:**

$$A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)$$

**Pure InfSA** replaces softmax with Frobenius-normalized ReLU, turning each layer into a spectral diffusion step:

$$\hat{A}^{(l)} = \frac{[\,Q^{(l)}{K^{(l)}}^\top\,]_+}{\|[\,Q^{(l)}{K^{(l)}}^\top\,]_+\|_F + \varepsilon}$$

Across $L$ Transformer layers, the accumulated output implements a truncated Neumann series:

$$S_L = \sum_{t=1}^{L} \gamma^{t}\, \hat{A}^{(t)} \cdots \hat{A}^{(1)} \, X^{(0)} \quad\xrightarrow{L\to\infty}\quad \bigl(I - \gamma\hat{A}\bigr)^{-1} - I$$

This is equivalent to computing **Katz centrality** on the token graph — the same mathematical object underlying **PageRank** and the **fundamental matrix of absorbing Markov chains**.

**Linear InfSA** approximates the principal eigenvector of the implicit attention operator in O(N) time:

$$\bar{q} = \sum_i \alpha_i Q_i, \qquad a_j = \frac{[\bar{q}^\top K_j]_+}{\sum_l [\bar{q}^\top K_l]_+ + \varepsilon}$$

<p align="center">
  <img src="figures/markov_higher_order.png" width="80%"/>
  <br>
  <em>Absorbing Markov chain interpretation: InfSA's multi-hop propagation correctly identifies globally important tokens that single-hop softmax attention misses.</em>
</p>

---

## API Reference

### `infsa.convert(model, variant, **kwargs)`

Replace all attention layers in any model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Model to convert |
| `variant` | `str` | `"pure_infsa"` | `"pure_infsa"` or `"linear_infsa"` |
| `rho_init` | `float` | `0.95` | Initial spectral decay ρ |
| `rho_trainable` | `bool` | `True` | Make ρ learnable |
| `copy_weights` | `bool` | `True` | Copy existing Q/K/V/O weights |
| `include_patterns` | `list[str]` | `None` | Regex: only convert matching module names |
| `exclude_patterns` | `list[str]` | `None` | Regex: skip matching module names |
| `target_types` | `list[Type]` | `None` | Only convert specific module types |
| `inplace` | `bool` | `False` | Modify model in place |

### `infsa.InfSAAttention(embed_dim, num_heads, variant, **kwargs)`

Drop-in `nn.Module` replacement for `nn.MultiheadAttention`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_dim` | `int` | — | Total embedding dimension |
| `num_heads` | `int` | — | Number of attention heads |
| `variant` | `str` | `"pure_infsa"` | `"pure_infsa"` or `"linear_infsa"` |
| `dropout` | `float` | `0.0` | Dropout on attention scores |
| `batch_first` | `bool` | `True` | Input shape `(B, N, E)` vs `(N, B, E)` |
| `rho_init` | `float` | `0.95` | Initial ρ value |
| `rho_trainable` | `bool` | `True` | Whether ρ adapts during training |

### `infsa.infsa_attention(q, k, v, variant, rho)`

Functional API on pre-projected tensors `(B, H, N, D)`.

### `infsa.pure_infsa_scores(q, k, rho)` / `infsa.linear_infsa_scores(q, k, rho)`

Compute raw attention scores without applying to values.

---

## Experimental Results

### Classification Accuracy

<p align="center">
  <img src="figures/accuracy_vs_params.png" width="85%"/>
  <br>
  <em>ImageNet-1K and ImageNet-V2 top-1 accuracy vs. parameter count. InfViT variants (red stars) achieve competitive or superior accuracy. On ImageNet-V2, all InfViT models exceed every baseline (up to 79.8% vs 76.8%).</em>
</p>

**Key results** (4-layer ViT, 53.5M params, DeiT recipe, no distillation):

| Model | IN-1K Top-1 | IN-V2 Top-1 | Params | GFLOPs |
|-------|------------|------------|--------|--------|
| Standard ViT 4L | 81.5% | — | 57.7M | 17.5 |
| **Linear InfViT 4L** | **84.7%** | **79.8%** | **53.5M** | 59 |
| Pure InfViT 4L | 85.1% | 79.5% | 57.7M | 59 |
| Pure InfViT 24L | 85.4% | 79.8% | 330.6M | — |

The +3.2 pp gain of Linear InfViT over Standard ViT is purely architectural — same training recipe, no external data.

### Efficiency & Scalability

<p align="center">
  <img src="figures/efficiency_dashboard.png" width="85%"/>
  <br>
  <em>Efficiency dashboard at 1024²: Linear InfViT achieves 13.4× speed-up, 231 img/s throughput, and 0.87 J/img — best in every metric.</em>
</p>

<p align="center">
  <img src="figures/complexity_tiers.png" width="85%"/>
  <br>
  <em>Throughput vs. energy per image for nine attention mechanisms, colored by asymptotic complexity. InfViT Linear (O(N), red star) dominates the top-left corner.</em>
</p>

| Metric | Standard ViT | Linear InfViT | Improvement |
|--------|-------------|---------------|-------------|
| Throughput (1024²) | 17.19 img/s | **230.95 img/s** | **13.4×** |
| Energy (1024²) | 11.63 J/img | **0.87 J/img** | **13.4×** |
| Max resolution | 1024² | **9216²** (332K tokens) | Only model to complete |
| Inference latency | 58.16 ms | **4.33 ms** | **13.4×** |

<p align="center">
  <img src="figures/cvpr_latency_train_infer_FINAL.png" width="85%"/>
  <br>
  <em>Training and inference latency across resolutions. Linear InfViT scales near-linearly to 9216² while all other models OOM beyond 1024².</em>
</p>

<details>
<summary><b>More scalability plots</b></summary>

<p align="center">
  <img src="figures/speedup_4L.png" width="45%"/>
  <img src="figures/throughput_4L.png" width="45%"/>
</p>

<p align="center">
  <img src="figures/energy_4L.png" width="45%"/>
  <img src="figures/latency_bars_4L.png" width="45%"/>
</p>

<p align="center">
  <img src="figures/depth_comparison_thr_energy.png" width="85%"/>
  <br>
  <em>4-layer vs 24-layer depth comparison: throughput and energy.</em>
</p>

</details>

### Attention Quality

<p align="center">
  <img src="figures/attention_quality_bars.png" width="85%"/>
  <br>
  <em>Attention quality: MoRF-AOC, ROC-AUC, and PR-AUC (%). InfSA variants outperform Standard ViT by 20–34 pp across all metrics.</em>
</p>

| Metric | Standard ViT | Pure InfSA | Linear InfSA |
|--------|-------------|-----------|-------------|
| MoRF-AOC ↑ | 42.6% | 71.7% | **76.0%** |
| ROC-AUC ↑ | 53.8% | **77.3%** | 75.9% |
| PR-AUC ↑ | 56.2% | 72.4% | **76.1%** |

<p align="center">
  <img src="figures/attention_bbox_combined.png" width="85%"/>
  <br>
  <em>Bounding-box localization: InfSA attention maps align precisely with object boundaries.</em>
</p>

<details>
<summary><b>More attention quality plots</b></summary>

<p align="center">
  <img src="figures/morf_lerf_combined.png" width="85%"/>
  <br>
  <em>MoRF degradation and LeRF retention curves.</em>
</p>

<p align="center">
  <img src="figures/morf_only.png" width="45%"/>
  <img src="figures/lerf_only.png" width="45%"/>
</p>

<p align="center">
  <img src="figures/roc_only.png" width="45%"/>
  <img src="figures/pr_only.png" width="45%"/>
</p>

<p align="center">
  <img src="figures/morf_lerf_curves_comparison.png" width="85%"/>
</p>

</details>

---

## Examples

| Example | Description |
|---------|-------------|
| [vit_torchvision.py](examples/vit_torchvision.py) | Convert a pretrained torchvision ViT-B/16 |
| [huggingface_vit.py](examples/huggingface_vit.py) | Convert a HuggingFace ViT |
| [huggingface_llm.py](examples/huggingface_llm.py) | InfSA in GPT-2 and custom LLMs |
| [custom_transformer.py](examples/custom_transformer.py) | Build a ViT from scratch with InfSA |

---

## Testing

### Unit tests

```bash
pytest tests/test_infsa.py -v
```

### Integration tests

End-to-end tests that validate InfSA on real model architectures with synthetic inputs:

| Test | Model | What it validates |
|------|-------|-------------------|
| `test_torchvision_vit.py` | torchvision ViT-B/16 | `infsa.convert()` on PyTorch ViT, inference + training + gradient flow |
| `test_huggingface_vit.py` | HuggingFace ViT | Custom wrapper for HF `ViTSelfAttention`, weight copying |
| `test_detr.py` | DETR (ResNet-50) | Monkey-patching DETR's cross/self-attention with InfSA functional API |
| `test_custom_transformer.py` | Custom ViT + MiniLLM | Building from scratch with `InfSAAttention`, `nn.TransformerEncoder` conversion |
| `test_qwen3_llm.py` | Qwen3-0.6B | Auto-converting a full LLM, inference on SQuAD/GSM8K |

Run all integration tests:

```bash
python tests/integration/run_all.py
```

Results are written to `results/`.

---

## Project Structure

```
InfSA_release/
├── infsa/                  # Core library (pip-installable)
│   ├── __init__.py         # Public API exports
│   ├── core.py             # Functional: pure_infsa_scores, linear_infsa_scores, infsa_attention
│   ├── attention.py        # nn.Module: InfSAAttention (drop-in for nn.MultiheadAttention)
│   └── convert.py          # Auto-conversion: convert(), replace_attention()
├── examples/               # Runnable usage examples
├── tests/
│   ├── test_infsa.py       # Unit tests (pytest)
│   └── integration/        # End-to-end integration tests
│       ├── helpers.py       # Shared test utilities
│       └── run_all.py       # Run all integration tests
├── datasets/               # Small sample datasets for integration tests
│   ├── imnet/              # ImageNet validation XML annotations
│   └── llm/                # LLM benchmark JSONL samples
├── figures/                # Paper figures used in this README
└── pyproject.toml          # Package metadata
```

---

## Citation

```bibtex
@article{roffo2026infsa,
  title={Self-Attention And Beyond the Infinite: Towards Linear Transformers with Infinite Self-Attention},
  author={Roffo, Giorgio and Palmer, Luke},
  journal={arXiv preprint arXiv:2603.00175},
  year={2026}
}
```

---

## Repository contents

- PyTorch implementation of InfSA
- PyTorch implementation of Linear-InfSA
- Vision Transformer experiments
- Training scripts
- Evaluation scripts
- Benchmarks on ImageNet / ImageNet-V2
- Long-resolution inference experiments

## Keywords

- Infinite Self-Attention
- InfSA
- Linear-InfSA
- linear attention
- self-attention
- transformer
- vision transformer
- efficient attention
- long-context / high-resolution inference

## Highlights
- Spectral formulation of self-attention
- Linear-InfSA: linear-time approximation
- Vision Transformer integration
- High-resolution inference support
- Benchmarks on ImageNet / ImageNet-V2

## License

MIT License. See [LICENSE](LICENSE).
