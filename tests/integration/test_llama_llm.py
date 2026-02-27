"""
Integration test: HuggingFace LLaMA 3.2-1B LLM with InfSA.

Loads a pretrained LLaMA 3.2-1B from HuggingFace, converts its attention layers
to InfSA, runs inference on the LLM benchmark datasets (SQuAD, GSM8K) from
datasets/llm/, and validates outputs + gradient flow.

Results are written to results/test_llama_llm.txt
"""

import sys
import os
import time
from datetime import datetime

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import infsa
from helpers import get_device, load_jsonl_samples, write_results

MODEL_NAME = "meta-llama/Llama-3.2-1B"


def extract_prompt(sample):
    """Extract the user prompt from a conversation-format sample."""
    if "conversations" in sample:
        for msg in sample["conversations"]:
            if msg["role"] == "user":
                return msg["content"]
    if "turns" in sample:
        return sample["turns"][0] if sample["turns"] else ""
    return str(sample)


def _get_device():
    return get_device()


def run_test():
    results = []
    results.append(f"Test: LLaMA 3.2-1B LLM + InfSA")
    results.append(f"Date: {datetime.now().isoformat()}")
    results.append(f"PyTorch: {torch.__version__}")
    results.append("=" * 60)

    device = _get_device()
    results.append(f"Device: {device}")

    all_passed = True

    # --- Load datasets ---
    datasets_to_test = [
        ("squad.jsonl", "SQuAD"),
        ("gsm8k.jsonl", "GSM8K"),
        ("math_reasoning.jsonl", "MathReasoning"),
    ]
    dataset_samples = {}
    for filename, label in datasets_to_test:
        try:
            samples = load_jsonl_samples(filename, max_samples=3)
            dataset_samples[label] = samples
            results.append(f"  [PASSED] Load {label} ({len(samples)} samples)")
        except Exception as e:
            results.append(f"  [FAILED] Load {label}: {e}")
            all_passed = False

    # --- Load tokenizer and model ---
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        results.append(f"  [PASSED] Tokenizer loaded ({MODEL_NAME})")
    except Exception as e:
        results.append(f"  [FAILED] Tokenizer loading: {e}")
        all_passed = False
        # Write and exit early
        _write_results(results, all_passed)
        return all_passed

    # --- Test both variants ---
    for variant in ["pure_infsa", "linear_infsa"]:
        results.append(f"\n--- Variant: {variant} ---")

        # 1. Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
            )
            model.eval()
            results.append(f"  [PASSED] Model loading ({MODEL_NAME})")
        except Exception as e:
            results.append(f"  [FAILED] Model loading: {e}")
            all_passed = False
            continue

        # 2. Convert to InfSA
        try:
            # Find the attention class used by LLaMA
            attn_types = set()
            for name, module in model.named_modules():
                cls_name = type(module).__name__
                if "attention" in cls_name.lower() and "layer" not in cls_name.lower():
                    attn_types.add(type(module))

            if attn_types:
                model = infsa.convert(
                    model,
                    variant=variant,
                    target_types=list(attn_types),
                    rho_init=0.9,
                    verbose=False,
                )
            else:
                # Fallback: auto-detection
                model = infsa.convert(
                    model,
                    variant=variant,
                    rho_init=0.9,
                    verbose=False,
                )

            infsa_count = sum(1 for _, m in model.named_modules() if isinstance(m, infsa.InfSAAttention))
            assert infsa_count > 0, "No InfSA layers found after conversion"
            results.append(f"  [PASSED] Conversion ({infsa_count} InfSA layers, target types: {[t.__name__ for t in attn_types]})")
        except Exception as e:
            results.append(f"  [FAILED] Conversion: {e}")
            all_passed = False
            continue

        model = model.to(device)

        # 3. Inference on dataset samples
        for label, samples in dataset_samples.items():
            try:
                model.eval()
                for i, sample in enumerate(samples[:2]):
                    prompt = extract_prompt(sample)
                    # Truncate long prompts for speed
                    prompt_truncated = prompt[:256]
                    inputs = tokenizer(
                        prompt_truncated,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128,
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        assert logits.ndim == 3, f"Unexpected logits ndim: {logits.ndim}"
                        seq_len = logits.shape[1]
                        vocab_size = logits.shape[2]

                    results.append(f"    {label} sample {i+1}: seq_len={seq_len}, vocab={vocab_size}")
                results.append(f"  [PASSED] Inference on {label}")
            except Exception as e:
                results.append(f"  [FAILED] Inference on {label}: {e}")
                all_passed = False

        # 4. Gradient flow
        try:
            model.train()
            dummy_ids = torch.randint(0, 1000, (1, 32), device=device)
            outputs = model(dummy_ids)
            loss = outputs.logits.sum()
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
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            dummy_ids = torch.randint(0, 1000, (1, 32), device=device)
            outputs = model(dummy_ids)
            loss = outputs.logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            results.append(f"  [PASSED] Training step (loss={loss.item():.4f})")
        except Exception as e:
            results.append(f"  [FAILED] Training step: {e}")
            all_passed = False

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Summary ---
    results.append("\n" + "=" * 60)
    status = "PASSED" if all_passed else "FAILED"
    results.append(f"OVERALL: {status}")

    _write_results(results, all_passed)
    return all_passed


def _write_results(results, all_passed):
    write_results(results, "test_llama_llm.txt")


if __name__ == "__main__":
    passed = run_test()
    sys.exit(0 if passed else 1)
