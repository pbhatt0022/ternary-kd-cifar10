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


SEEDS_EXPECTED = {"T": 1, "A": 3, "B": 2, "C": 3, "D": 3, "E": 1, "D_T2": 2, "D_T8": 2}


def completed_runs(runs_dir):
    """Runs eligible for test evaluation, grouped by arm, in a stable order.

    Section 2 allows the test set only once an arm's training is COMPLETE, so an arm with
    fewer completed runs than its seed complement is held back entirely. Without this, a
    Phase 4 run triggered while Phase 3 is still going would read the test set early for a
    half-finished arm -- which cannot be undone once done.
    """
    by_group = {}
    for manifest_path in sorted(pathlib.Path(runs_dir).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("discard_status") != "completed":
            continue
        label = manifest_path.parent.name
        by_group.setdefault(label.rsplit("_", 1)[0], []).append(label)

    eligible, held_back = [], []
    for group, labels in sorted(by_group.items()):
        needed = SEEDS_EXPECTED.get(group)
        if needed is not None and len(labels) < needed:
            held_back.append(f"{group} ({len(labels)}/{needed})")
            continue
        eligible.extend(sorted(labels))

    if held_back:
        print("held back, arm not finished: " + ", ".join(held_back))
    return eligible


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
