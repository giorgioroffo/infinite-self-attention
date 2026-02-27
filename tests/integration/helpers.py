"""
Shared utilities for InfSA integration tests.

Provides common functions for device detection, dataset loading,
and result writing used across all integration test scripts.
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
IMNET_DIR = PROJECT_ROOT / "datasets" / "imnet"
LLM_DIR = PROJECT_ROOT / "datasets" / "llm"

# ─── Load .env ────────────────────────────────────────────────────────────────

def load_dotenv():
    """Load environment variables from .env file at project root."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                os.environ.setdefault(key, value)

load_dotenv()


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return cuda device if CUBLAS works, else cpu."""
    if torch.cuda.is_available():
        try:
            t = torch.randn(64, 64, device="cuda")
            _ = t @ t
            torch.cuda.synchronize()
            del t
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


# ─── Dataset loaders ──────────────────────────────────────────────────────────

def load_imnet_annotations(include_bbox=False):
    """Load ImageNet annotation XMLs from datasets/imnet/.

    Args:
        include_bbox: If True, include bounding box and class_id fields.
    """
    annotations = []
    for xml_file in sorted(IMNET_DIR.glob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        entry = {
            "filename": root.findtext("filename"),
            "width": int(root.findtext("size/width")),
            "height": int(root.findtext("size/height")),
        }
        if include_bbox:
            obj = root.find("object")
            entry["class_id"] = obj.findtext("name") if obj is not None else "unknown"
            bndbox = obj.find("bndbox") if obj is not None else None
            if bndbox is not None:
                entry["bbox"] = [int(bndbox.findtext(k)) for k in ["xmin", "ymin", "xmax", "ymax"]]
            else:
                entry["bbox"] = None
        annotations.append(entry)
    return annotations


def load_jsonl_samples(filename, max_samples=3):
    """Load samples from a JSONL dataset file in datasets/llm/."""
    filepath = LLM_DIR / filename
    samples = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line.strip()))
    return samples


# ─── Result writer ────────────────────────────────────────────────────────────

def write_results(results, output_filename):
    """Write result lines to a text file in results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / output_filename
    output_path.write_text("\n".join(results) + "\n")
    print("\n".join(results))
