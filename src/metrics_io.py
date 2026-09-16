"""Reading metrics.jsonl. Writers append only; duplicates are resolved on read.

metrics.jsonl lives on a network filesystem (Google Drive) and is appended once per epoch.
An earlier version rewrote the file on resume to drop epochs past the resume point. Drive
served a stale copy of the file to that read-modify-write, and the rewrite made the
staleness permanent, destroying three epochs of one run.

A resumed run can therefore re-log an epoch it had already logged, which is harmless: the
later record supersedes the earlier one, and keeping the last record per epoch is equivalent
to truncation without ever destroying data.
"""

import json


def read_metrics(path):
    """Parsed per-epoch records, one per epoch, ordered by epoch.

    Where an epoch appears more than once the last occurrence wins, since that is the one a
    resumed run actually produced.
    """
    by_epoch = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        by_epoch[record["epoch"]] = record
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def val_accuracies(path, epochs):
    """Validation accuracy for the given epochs, skipping any that are absent."""
    by_epoch = {record["epoch"]: record["val_accuracy"] for record in read_metrics(path)}
    return [by_epoch[epoch] for epoch in epochs if epoch in by_epoch]
