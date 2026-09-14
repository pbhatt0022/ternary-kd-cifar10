"""Pre-flight verification. Pre-registration section 9.

These are the checks that need real hardware or the real training loop, as opposed to the
pure-logic unit tests in tests/. Phase 0 of the orchestration runs both and aborts on any
failure.

    python scripts/verify.py --data-dir /content/data
    python scripts/verify.py --restore-check --runs-dir <dir>   # after the teacher trains

Exits non-zero if anything fails, so a notebook cell goes red rather than quietly
continuing into training.
"""

import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from src import train
from src.data import split_indices
from src.models import EXPECTED_TERNARY_CONVS, resnet18_cifar, resnet34_cifar
from src.ternary_modules import TernaryConv2d, ternarize_model

REPO = pathlib.Path(__file__).resolve().parents[1]
DETERMINISM_EPOCHS = 2
DETERMINISM_ARM = "C"  # ternary: exercises the quantizer and the STE, not just the loop

_results = []


def check(name):
    """Decorator registering a check. The function returns None on pass or a reason."""

    def wrap(fn):
        _results.append((name, fn))
        return fn

    return wrap


@check("split is byte-identical and disjoint")
def _split():
    train_idx, val_idx = split_indices()  # raises if the hash drifted
    if len(train_idx) != 45_000 or len(val_idx) != 5_000:
        return f"sizes {len(train_idx)}/{len(val_idx)}, expected 45000/5000"
    if set(train_idx) & set(val_idx):
        return "train and validation overlap"
    return None


@check("determinism and TF32 flags set correctly")
def _flags():
    train.set_determinism(0)
    if not torch.backends.cudnn.deterministic:
        return "cudnn.deterministic is False"
    if torch.backends.cudnn.benchmark:
        return "cudnn.benchmark is True"
    if torch.backends.cudnn.allow_tf32:
        return "cudnn TF32 is enabled; the pre-registration is fp32 only"
    if torch.backends.cuda.matmul.allow_tf32:
        return "matmul TF32 is enabled; the pre-registration is fp32 only"
    return None


@check("LR schedule boundaries are exact")
def _lr_schedule():
    expected = {1: 0.1, 79: 0.1, 80: 0.01, 119: 0.01, 120: 0.001, 160: 0.001}
    for epoch, want in expected.items():
        got = train.lr_for_epoch(epoch)
        if abs(got - want) > 1e-12:
            return f"epoch {epoch}: lr={got}, expected {want}"
    return None


@check("ternarized layer set is exactly as specified")
def _ternary_set():
    model = resnet18_cifar()
    names = ternarize_model(model, expected_count=EXPECTED_TERNARY_CONVS["resnet18_cifar"])
    if len(names) != 19:
        return f"{len(names)} ternarized convs, expected 19"
    if isinstance(model.conv1, TernaryConv2d):
        return "the stem conv1 was ternarized; it must stay fp32"
    if type(model.fc) is not torch.nn.Linear:
        return "the final FC must stay a full-precision nn.Linear"
    if sum(1 for n in names if n.endswith("downsample.0")) != 3:
        return "the three 1x1 downsample convs must be ternarized"
    return None


@check("weight decay excludes BatchNorm parameters and biases")
def _weight_decay_groups():
    model = resnet18_cifar()
    groups = train.param_groups(model)
    decayed = {id(p) for p in groups[0]["params"]}
    if groups[0]["weight_decay"] != train.WEIGHT_DECAY or groups[1]["weight_decay"] != 0.0:
        return "param group weight decay values are wrong"
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if id(module.weight) in decayed or id(module.bias) in decayed:
                return "a BatchNorm parameter is in the decayed group"
    if id(model.fc.bias) in decayed:
        return "the FC bias is in the decayed group"
    if id(model.fc.weight) not in decayed:
        return "the FC weight must be decayed"
    return None


@check("src/train.py never touches the test set")
def _data_hygiene():
    source = (REPO / "src" / "train.py").read_text(encoding="utf-8")
    for needle in ("test_loader", "train=False"):
        if needle in source:
            return f"src/train.py references {needle!r}"
    return None


def _short_run(seed, scratch, data_dir):
    """A fresh DETERMINISM_EPOCHS-epoch run, returning its per-epoch records."""
    if scratch.exists():
        shutil.rmtree(scratch)  # a leftover resume.pt would resume instead of restart
    scratch.mkdir(parents=True)
    train.run(
        DETERMINISM_ARM, seed, scratch, data_dir,
        label="determinism", max_epochs=DETERMINISM_EPOCHS,
    )
    lines = (scratch / "determinism" / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(l) for l in lines.splitlines() if l.strip()]


def check_determinism(runs_dir, data_dir):
    """Two runs at the same seed must agree exactly; a different seed must not.

    Section 9 asks for equality of the step-1 loss, the end-of-epoch-1 loss and the
    epoch-2 validation accuracy. Comparing the epoch-1 MEAN loss is strictly stronger
    than comparing step 1 alone, since the mean is a function of every step in the epoch
    and so diverges if any step does.
    """
    scratch = pathlib.Path(runs_dir) / "_verify"
    keys = ("train_loss_mean", "train_loss_final_step", "val_accuracy")

    first = _short_run(0, scratch, data_dir)
    second = _short_run(0, scratch, data_dir)
    other = _short_run(1, scratch, data_dir)
    shutil.rmtree(scratch, ignore_errors=True)

    for epoch_index, (a, b) in enumerate(zip(first, second), start=1):
        for key in keys:
            if a[key] != b[key]:
                return f"same seed diverged at epoch {epoch_index} on {key}: {a[key]} vs {b[key]}"

    if all(first[i][k] == other[i][k] for i in range(len(first)) for k in keys):
        return "seed 0 and seed 1 produced identical results; the seed is not wired in"
    return None


def check_restore_reproduce(runs_dir, data_dir):
    """Reload a teacher checkpoint and reproduce its logged VALIDATION accuracy exactly.

    Test is not involved. Runs after the teacher trains, so it is not part of Phase 0.
    """
    run_dir = pathlib.Path(runs_dir) / train.TEACHER_RUN
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return f"no metrics at {metrics_path}; train arm T first"

    logged = {
        json.loads(l)["epoch"]: json.loads(l)["val_accuracy"]
        for l in metrics_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    if not checkpoints:
        return "no checkpoints found for the teacher run"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train.set_determinism(0)
    _, val_loader = train.train_val_loaders(0, root=data_dir, batch_size=train.BATCH_SIZE)

    path = checkpoints[0]
    epoch = int(path.stem.split("_")[1])
    model = resnet34_cifar().to(device)
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    recomputed = train.accuracy(model, val_loader, device)

    if epoch not in logged:
        return f"epoch {epoch} is checkpointed but absent from metrics.jsonl"
    if recomputed != logged[epoch]:
        return (
            f"epoch {epoch}: reloaded checkpoint gives {recomputed}, "
            f"metrics.jsonl logged {logged[epoch]}"
        )
    print(f"    epoch {epoch} reproduced exactly at {recomputed}% validation")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--runs-dir", default="./runs")
    parser.add_argument("--skip-determinism", action="store_true",
                        help="skip the two short training runs (they need a GPU)")
    parser.add_argument("--restore-check", action="store_true",
                        help="run only the teacher restore-and-reproduce check")
    args = parser.parse_args()

    if args.restore_check:
        reason = check_restore_reproduce(args.runs_dir, args.data_dir)
        print(("FAIL  " if reason else "ok    ") + "restore-and-reproduce"
              + (f"\n      {reason}" if reason else ""))
        return 1 if reason else 0

    failures = 0
    for name, fn in _results:
        reason = fn()
        print(("FAIL  " if reason else "ok    ") + name + (f"\n      {reason}" if reason else ""))
        failures += bool(reason)

    if args.skip_determinism:
        print("skip  determinism (--skip-determinism)")
    else:
        print(f"...   determinism: {3 * DETERMINISM_EPOCHS} short epochs, please wait")
        reason = check_determinism(args.runs_dir, args.data_dir)
        print(("FAIL  " if reason else "ok    ") + "determinism"
              + (f"\n      {reason}" if reason else ""))
        failures += bool(reason)

    print()
    if failures:
        print(f"{failures} check(s) FAILED -- do not start training")
        return 1
    print("all pre-flight checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
