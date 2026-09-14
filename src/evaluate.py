"""End-of-arm test evaluation. Pre-registration section 7.

The ONLY place the test set is read. Runs after an arm's training is complete, from saved
checkpoints. The primary metric is the arithmetic mean of ten separately computed test
accuracies -- never an average of weights (section 12 forbids weight averaging).
"""

import json
import pathlib

import torch

from src.data import test_loader
from src.train import BATCH_SIZE, CHECKPOINT_EPOCHS, accuracy, build_model, set_determinism

SETTLEDNESS_THRESHOLD = 0.5  # percentage points, a declared round number


def _val_spread(metrics_path, epochs):
    """max - min of validation accuracy over `epochs`, in percentage points."""
    records = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["epoch"]] = record["val_accuracy"]
    values = [records[e] for e in epochs if e in records]
    if len(values) != len(list(epochs)):
        raise ValueError(
            f"{metrics_path}: found {len(values)} of {len(list(epochs))} final-window epochs"
        )
    return max(values) - min(values)


def evaluate_run(runs_dir, label, data_dir, device=None, loader=None):
    """Evaluate one run's ten final checkpoints on the test set and write test_results.json.

    Returns the results dict. Idempotent: if test_results.json already exists it is read
    back rather than recomputed, so re-running Phase 4 costs nothing and cannot change a
    number that has already been reported.
    """
    run_dir = pathlib.Path(runs_dir) / label
    results_path = run_dir / "test_results.json"
    if results_path.exists():
        return json.loads(results_path.read_text(encoding="utf-8"))

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["discard_status"] != "completed":
        raise ValueError(f"{label} is {manifest['discard_status']}; not evaluable")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(manifest["seed"])
    model, _ = build_model(manifest["arm"], device)
    if loader is None:
        loader = test_loader(root=data_dir, batch_size=BATCH_SIZE)

    accuracies = []
    for epoch in CHECKPOINT_EPOCHS:
        path = run_dir / "checkpoints" / f"epoch_{epoch}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint {path}")
        # Ternary arms store fp32 latent weights; requantization at load is deterministic,
        # so the forward pass reproduces exactly what training saw.
        model.load_state_dict(torch.load(path, map_location=device)["model"])
        accuracies.append(accuracy(model, loader, device))

    spread = _val_spread(run_dir / "metrics.jsonl", CHECKPOINT_EPOCHS)
    results = {
        "label": label,
        "arm": manifest["arm"],
        "seed": manifest["seed"],
        "per_checkpoint_test_accuracy": accuracies,
        "last10_mean": sum(accuracies) / len(accuracies),
        "epoch160_accuracy": accuracies[-1],
        "last10_val_spread": spread,
        "settledness_flag": spread > SETTLEDNESS_THRESHOLD,
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
