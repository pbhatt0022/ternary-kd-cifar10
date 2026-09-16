"""Training-set accuracy of each run's final checkpoint, for the train/test gap.

The assignment asks for a teacher with a small train/test gap. Training accuracy was not part of
the pre-registered per-epoch log, so it is computed here afterwards, from the epoch-160
checkpoint, on the 45k training split with augmentation OFF (the same transform validation and
test use). It is descriptive only: it reads neither the test set nor anything used for selection,
and it changes no reported number.

    python scripts/train_accuracy.py --runs-dir RUNS --data-dir DATA
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.data import NUM_WORKERS, _eval_transform, split_indices
from src.train import BATCH_SIZE, accuracy, build_model, set_determinism

HEADLINE = {"T": 1, "A": 3, "B": 2, "C": 3, "D": 3, "E": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epoch", type=int, default=160)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_idx, _ = split_indices()
    loader = DataLoader(
        Subset(datasets.CIFAR10(args.data_dir, train=True, download=True,
                                transform=_eval_transform()), train_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"{'run':<7}{'train acc':>10}{'test ep160':>13}{'gap (pp)':>10}")
    for arm, seeds in HEADLINE.items():
        for seed in range(seeds):
            run_dir = pathlib.Path(args.runs_dir) / f"{arm}_{seed}"
            ckpt = run_dir / "checkpoints" / f"epoch_{args.epoch}.pt"
            if not ckpt.exists():
                print(f"{run_dir.name:<7} missing {ckpt.name}")
                continue
            set_determinism(seed)
            model, _ = build_model(arm, device)
            model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
            train_acc = accuracy(model, loader, device)

            test_path = run_dir / "test_results.json"
            test = (json.loads(test_path.read_text(encoding="utf-8"))["epoch160_accuracy"]
                    if test_path.exists() else None)
            gap = None if test is None else train_acc - test
            (run_dir / "train_accuracy.json").write_text(json.dumps({
                "epoch": args.epoch,
                "train_accuracy_no_augmentation": train_acc,
                "test_accuracy_same_epoch": test,
                "train_minus_test_pp": gap,
            }, indent=2), encoding="utf-8")
            print(f"{run_dir.name:<7}{train_acc:>9.2f}%"
                  f"{'-' if test is None else f'{test:.2f}%':>13}"
                  f"{'-' if gap is None else f'{gap:.2f}':>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
