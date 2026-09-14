"""Phase 4: evaluate every completed run on the test set. Pre-registration section 7.

This is the first and only test-set access in the whole study. It runs after training, from
saved checkpoints, and writes one test_results.json per run.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from src.data import test_loader
from src.evaluate import evaluate_run
from src.train import BATCH_SIZE


def completed_runs(runs_dir):
    """Every run directory whose manifest says it finished cleanly, in a stable order."""
    labels = []
    for manifest_path in sorted(pathlib.Path(runs_dir).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("discard_status") == "completed":
            labels.append(manifest_path.parent.name)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    labels = completed_runs(args.runs_dir)
    if not labels:
        print("no completed runs to evaluate")
        return 1

    # One test loader, reused: building it per run would re-read the dataset needlessly.
    loader = test_loader(root=args.data_dir, batch_size=BATCH_SIZE)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"evaluating {len(labels)} run(s) on the test set\n")
    failures = 0
    for label in labels:
        try:
            results = evaluate_run(args.runs_dir, label, args.data_dir,
                                   device=device, loader=loader)
        except (FileNotFoundError, ValueError) as error:
            print(f"FAIL  {label}: {error}")
            failures += 1
            continue
        flag = "  UNSETTLED" if results["settledness_flag"] else ""
        print(
            f"ok    {label:<10} last10={results['last10_mean']:.2f}%  "
            f"epoch160={results['epoch160_accuracy']:.2f}%  "
            f"val_spread={results['last10_val_spread']:.2f}pp{flag}"
        )

    print()
    if failures:
        print(f"{failures} run(s) could not be evaluated")
        return 1
    print("all completed runs evaluated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
