"""CIFAR-10 loading and the fixed 45k/5k split. Pre-registration §2.

The split is a property of the study, not of a run: it is derived from SPLIT_SEED
alone and verified against SPLIT_HASH on every call, so drift is a hard error
rather than a silent change in what "validation" means.
"""

import hashlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SPLIT_SEED = 20260828  # study-wide; NOT the run seed
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

N_TRAIN_FULL = 50_000
N_TRAIN = 45_000
N_VAL = 5_000

# sha256 over train_idx.tobytes() + val_idx.tobytes(), int64, in permutation order
# (NOT sorted -- the order fixes which image lands in which batch under a seeded sampler).
SPLIT_HASH = "014b3a6e8b9a41124c1375c3a520393439febfad740f32bacde52c79ebddf445"

# prereg §5 fixes num_workers=4. Do not "tune" this: the worker count partitions the
# augmentation RNG stream, so changing it changes every augmented batch.
NUM_WORKERS = 4


def split_indices():
    """Return (train_idx, val_idx) as int64 arrays. Identical for every run and arm."""
    perm = np.random.RandomState(SPLIT_SEED).permutation(N_TRAIN_FULL)
    train_idx = perm[:N_TRAIN].astype(np.int64)
    val_idx = perm[N_TRAIN:].astype(np.int64)
    digest = hashlib.sha256(train_idx.tobytes() + val_idx.tobytes()).hexdigest()
    if digest != SPLIT_HASH:
        raise RuntimeError(
            f"45k/5k split drifted: got {digest}, expected {SPLIT_HASH}. "
            "Refusing to train -- every arm must see the same split."
        )
    return train_idx, val_idx


def _eval_transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )


def _train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # zero padding
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def _worker_init_fn(seed):
    def init(worker_id):
        s = seed + worker_id
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)  # torchvision transforms draw from the torch RNG

    return init


def train_val_loaders(seed, root="./data", batch_size=128, download=True):
    """Training and validation loaders. Validation is augmentation-free.

    `seed` is the run seed: it controls shuffling and augmentation only. The split
    itself is fixed at SPLIT_SEED.
    """
    train_idx, val_idx = split_indices()
    augmented = datasets.CIFAR10(root, train=True, download=download, transform=_train_transform())
    plain = datasets.CIFAR10(root, train=True, download=download, transform=_eval_transform())

    train_loader = DataLoader(
        Subset(augmented, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=_worker_init_fn(seed),
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(plain, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def test_loader(root="./data", batch_size=128, download=True):
    """CIFAR-10 test set. Called ONLY from src/evaluate.py, after an arm finishes.

    prereg §2: no component of the design reads test accuracy during training.
    scripts/verify.py greps src/train.py to enforce that this is never called there.
    """
    return DataLoader(
        datasets.CIFAR10(root, train=False, download=download, transform=_eval_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
