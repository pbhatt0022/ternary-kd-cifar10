"""Reading metrics.jsonl. Torch-free, so it runs anywhere.

These cases exist because a destructive read-modify-write on this file, on a network
filesystem that served a stale copy, permanently lost three epochs of one run. Writers now
append only and readers resolve duplicates.
"""

import json

from src.metrics_io import read_metrics, val_accuracies


def _write(tmp_path, records):
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_reads_in_epoch_order(tmp_path):
    path = _write(tmp_path, [{"epoch": e, "val_accuracy": 90.0 + e} for e in (1, 2, 3)])
    assert [r["epoch"] for r in read_metrics(path)] == [1, 2, 3]


def test_last_record_wins_for_a_repeated_epoch():
    """A resumed run may re-log an epoch. The later record is the one it produced."""
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as d:
        path = _write(
            pathlib.Path(d),
            [
                {"epoch": 1, "val_accuracy": 10.0},
                {"epoch": 2, "val_accuracy": 20.0},
                {"epoch": 2, "val_accuracy": 22.0},
                {"epoch": 3, "val_accuracy": 30.0},
            ],
        )
        records = read_metrics(path)
        assert [r["epoch"] for r in records] == [1, 2, 3]
        assert records[1]["val_accuracy"] == 22.0


def test_blank_lines_ignored(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"epoch": 1, "val_accuracy": 1.0}) + "\n\n"
        + json.dumps({"epoch": 2, "val_accuracy": 2.0}) + "\n",
        encoding="utf-8",
    )
    assert len(read_metrics(path)) == 2


def test_val_accuracies_skips_absent_epochs(tmp_path):
    """An interior gap must not shift or fabricate values -- it yields a shorter list, and
    callers that need a complete window check the length themselves."""
    path = _write(
        tmp_path,
        [{"epoch": e, "val_accuracy": float(e)} for e in (1, 2, 5)],
    )
    assert val_accuracies(path, range(1, 6)) == [1.0, 2.0, 5.0]
    assert val_accuracies(path, [3, 4]) == []


def test_out_of_order_lines_are_sorted(tmp_path):
    path = _write(
        tmp_path,
        [{"epoch": 3, "val_accuracy": 3.0}, {"epoch": 1, "val_accuracy": 1.0}],
    )
    assert [r["epoch"] for r in read_metrics(path)] == [1, 3]
