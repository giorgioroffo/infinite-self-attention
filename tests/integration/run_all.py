"""
Run all InfSA integration tests and collect results.

Each test processes inputs from datasets/ and writes a result file
to results/ indicating PASSED or FAILED.

Usage:
    python tests/integration/run_all.py
"""

import sys
import subprocess
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TESTS_DIR.parent.parent / "results"

TEST_SCRIPTS = [
    "test_torchvision_vit.py",
    "test_huggingface_vit.py",
    "test_custom_transformer.py",
    "test_detr.py",
    "test_llama_llm.py",
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    overall_passed = True

    print("=" * 70)
    print("InfSA Integration Test Suite")
    print("=" * 70)

    for script in TEST_SCRIPTS:
        script_path = TESTS_DIR / script
        if not script_path.exists():
            print(f"\n[SKIP] {script} — not found")
            summary.append(f"[SKIP] {script}")
            continue

        print(f"\n{'─' * 70}")
        print(f"Running: {script}")
        print(f"{'─' * 70}")

        start = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            cwd=str(TESTS_DIR.parent.parent),
        )
        elapsed = time.time() - start

        status = "PASSED" if result.returncode == 0 else "FAILED"
        if status == "FAILED":
            overall_passed = False
        summary.append(f"[{status}] {script} ({elapsed:.1f}s)")

    # Write summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for line in summary:
        print(f"  {line}")

    overall = "PASSED" if overall_passed else "FAILED"
    print(f"\nOVERALL: {overall}")

    summary_path = RESULTS_DIR / "test_summary.txt"
    summary_path.write_text("\n".join(summary) + f"\n\nOVERALL: {overall}\n")
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
