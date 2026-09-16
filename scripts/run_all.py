"""Orchestration. Pre-registration sections 5, 6 and 10.

The only module that knows the phase ordering. Idempotent by design: it decides what to do
by reading runs/ rather than by remembering anything, so re-running it after a Colab
disconnect resumes exactly where it stopped (amendment A3).

    python scripts/run_all.py --runs-dir <dir> --data-dir <dir>
    python scripts/run_all.py --runs-dir <dir> --data-dir <dir> --phase 3
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import train
from src.metrics_io import val_accuracies as metrics_val_accuracies

REPO = pathlib.Path(__file__).resolve().parents[1]

TEACHER = ("T", 0)
TEACHER_GATE_VAL_ACCURACY = 92.0
HEADLINE_ARMS = {"A": 3, "B": 2, "C": 3, "D": 3, "E": 1}  # arm -> completed runs required
SWEEP_TEMPERATURES = (2.0, 8.0)  # T=4 is already covered by arm D seeds 0 and 1
SWEEP_SEEDS = (0, 1)
SEED_SEQUENCE = tuple(range(16))
MAX_DISCARDS = 3
LAST_10 = range(151, 161)


def status_of(runs_dir, label):
    """The run's terminal discard_status, or None if it never finished."""
    path = pathlib.Path(runs_dir) / label / "manifest.json"
    if not path.exists():
        return None
    status = json.loads(path.read_text(encoding="utf-8")).get("discard_status")
    return None if status == "running" else status


def val_accuracies(runs_dir, label, epochs):
    path = pathlib.Path(runs_dir) / label / "metrics.jsonl"
    return metrics_val_accuracies(path, epochs)


def _code_fingerprint():
    """Hash of everything Phase 0 actually verifies: src/ plus the verify script itself.

    Keyed on content rather than the commit hash so that edits to the notebook, the README
    or the orchestrator do not force the six determinism epochs to run again, while any
    change to training or to the checks does.
    """
    digest = hashlib.sha256()
    paths = sorted((REPO / "src").glob("*.py")) + [REPO / "scripts" / "verify.py"]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=REPO).returncode


def phase0(args):
    """Pre-flight. Nothing trains until this passes.

    The determinism check trains six short epochs, so the pass is recorded against a
    fingerprint of the code it verified: a resumed session skips it, and any change to
    src/ or to verify.py forces it to run again.
    """
    print("\n=== Phase 0: pre-flight ===")
    if args.skip_verify:
        print("skipped by --skip-verify")
        return 0

    marker = pathlib.Path(args.runs_dir) / ".preflight_passed"
    fingerprint = _code_fingerprint()
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == fingerprint:
        print(f"already passed for this code state ({fingerprint[:8]})")
        return 0

    if _run([sys.executable, "-m", "pytest", "tests", "-q"]):
        print("unit tests FAILED")
        return 1
    code = _run([
        sys.executable, "scripts/verify.py",
        "--data-dir", args.data_dir, "--runs-dir", args.runs_dir,
    ])
    if code == 0:
        marker.write_text(fingerprint, encoding="utf-8")
    return code


def phase1(args):
    """Arm E's architecture is already committed; confirm the code still agrees.

    The decision itself was made and committed before any training, which is what
    section 6 requires -- this phase cannot change it, only detect drift.
    """
    print("\n=== Phase 1: arm E selection ===")
    return _run([sys.executable, "scripts/select_arm_e.py", "--check"])


def phase2(args):
    """The teacher, then its gate, then restore-and-reproduce."""
    print("\n=== Phase 2: teacher ===")
    arm, seed = TEACHER
    label = f"{arm}_{seed}"
    status = status_of(args.runs_dir, label)

    if status is None:
        status = train.run(arm, seed, args.runs_dir, args.data_dir)["discard_status"]
    else:
        print(f"{label} already finished ({status})")

    if status != "completed":
        print(f"HALT: the teacher run is {status}. Investigate before continuing.")
        return 1

    accuracies = val_accuracies(args.runs_dir, label, LAST_10)
    if len(accuracies) != 10:
        print(f"HALT: expected 10 last-10 validation accuracies, found {len(accuracies)}")
        return 1
    mean = sum(accuracies) / len(accuracies)
    print(f"teacher last-10 mean validation accuracy: {mean:.2f}%")

    # Recorded so Phase 3 can refuse to start on an ungated teacher, which is what makes
    # queueing the phases back to back safe.
    gate_path = pathlib.Path(args.runs_dir) / "teacher_gate.json"
    gate_path.write_text(
        json.dumps({
            "last10_mean_val_accuracy": mean,
            "threshold": TEACHER_GATE_VAL_ACCURACY,
            "passed": mean >= TEACHER_GATE_VAL_ACCURACY,
            "overridden": bool(args.skip_gate),
        }, indent=2),
        encoding="utf-8",
    )

    if mean < TEACHER_GATE_VAL_ACCURACY:
        print(
            f"HALT: below the {TEACHER_GATE_VAL_ACCURACY}% gate. Section 6 treats this as a\n"
            "pipeline bug to investigate. Do NOT reconfigure, retune or reseed the teacher\n"
            "to pass. If the pipeline is genuinely correct, record the weak teacher as a\n"
            "limitation and re-run with --skip-gate."
        )
        if not args.skip_gate:
            return 1

    print("\n--- restore-and-reproduce ---")
    return _run([
        sys.executable, "scripts/verify.py", "--restore-check",
        "--runs-dir", args.runs_dir, "--data-dir", args.data_dir,
    ])


def run_arm(arm, needed, args, temperature=None, label_prefix=None):
    """Train an arm until `needed` runs complete, drawing seeds in order on discard.

    Section 10: replacements are drawn strictly in sequence, never chosen, and an arm that
    exceeds MAX_DISCARDS is reported as unstable rather than reseeded indefinitely.
    """
    prefix = label_prefix or arm
    completed, discarded = [], []

    for seed in SEED_SEQUENCE:
        if len(completed) >= needed:
            break
        if len(discarded) > MAX_DISCARDS:
            print(
                f"HALT: arm {prefix} exceeded {MAX_DISCARDS} discards "
                f"({', '.join(discarded)}). Report the configuration as unstable for this "
                "arm rather than drawing more seeds."
            )
            return None

        label = f"{prefix}_{seed}"
        status = status_of(args.runs_dir, label)
        if status is None:
            kwargs = {"label": label}
            if temperature is not None:
                kwargs["temperature"] = temperature
            status = train.run(arm, seed, args.runs_dir, args.data_dir, **kwargs)[
                "discard_status"
            ]
        else:
            print(f"{label} already finished ({status})")

        (completed if status == "completed" else discarded).append(label)

    if len(completed) < needed:
        print(f"HALT: arm {prefix} has {len(completed)}/{needed} completed runs and ran "
              "out of seeds")
        return None
    return {"completed": completed, "discarded": discarded}


def phase3(args):
    print("\n=== Phase 3: headline arms ===")

    # Section 6: the study does not proceed on a teacher that failed its gate. Checked here
    # rather than trusting phase order, so Phase 3 can be queued behind Phase 2 overnight
    # and will still stop itself if the teacher came out wrong.
    teacher_label = f"{TEACHER[0]}_{TEACHER[1]}"
    if status_of(args.runs_dir, teacher_label) != "completed":
        print(f"HALT: {teacher_label} has not completed. Phase 2 must finish first.")
        return 1

    # Recomputed from metrics.jsonl rather than read from teacher_gate.json, so this works
    # regardless of which code version trained the teacher.
    accuracies = val_accuracies(args.runs_dir, teacher_label, LAST_10)
    if len(accuracies) != 10:
        print(f"HALT: {teacher_label} has {len(accuracies)} of 10 final-window epochs")
        return 1
    mean = sum(accuracies) / len(accuracies)
    gate_path = pathlib.Path(args.runs_dir) / "teacher_gate.json"
    overridden = args.skip_gate or (
        gate_path.exists()
        and json.loads(gate_path.read_text(encoding="utf-8")).get("overridden", False)
    )
    if mean < TEACHER_GATE_VAL_ACCURACY and not overridden:
        print(
            f"HALT: the teacher scored {mean:.2f}% against a "
            f"{TEACHER_GATE_VAL_ACCURACY}% gate, and the failure has not been\n"
            "investigated. Section 6 treats this as a pipeline bug. Do not train the\n"
            "headline arms against a teacher that has not been accounted for.\n"
            "Re-run with --skip-gate only after investigating and recording it."
        )
        return 1
    print(f"teacher gate: {mean:.2f}% "
          f"({'passed' if mean >= TEACHER_GATE_VAL_ACCURACY else 'OVERRIDDEN'})")

    summary = {}
    for arm, needed in HEADLINE_ARMS.items():
        result = run_arm(arm, needed, args)
        if result is None:
            return 1
        summary[arm] = result
        print(f"arm {arm}: {len(result['completed'])} completed, "
              f"{len(result['discarded'])} discarded")
    _write_discards(args.runs_dir, summary)
    return 0


def phase4(args):
    """The first and only test-set access."""
    print("\n=== Phase 4: evaluation ===")
    script = REPO / "scripts" / "evaluate_all.py"
    if not script.exists():
        print("scripts/evaluate_all.py is not written yet; stopping before Phase 4.")
        return 1
    return _run([
        sys.executable, str(script),
        "--runs-dir", args.runs_dir, "--data-dir", args.data_dir,
    ])


def phase5(args):
    """The sweep runs only after Phase 4's numbers exist, and cannot revise them."""
    print("\n=== Phase 5: temperature sweep ===")
    for arm, needed in HEADLINE_ARMS.items():
        for seed in SEED_SEQUENCE[:needed]:
            results = pathlib.Path(args.runs_dir) / f"{arm}_{seed}" / "test_results.json"
            if not results.exists():
                print(f"HALT: {results} missing. Phase 4 must complete before the sweep.")
                return 1

    # --sweep-temperature lets two GPU notebooks split the sweep, one temperature each, so
    # they write to disjoint run directories and never contend for the same run.
    temperatures = SWEEP_TEMPERATURES
    if args.sweep_temperature is not None:
        if args.sweep_temperature not in SWEEP_TEMPERATURES:
            print(f"HALT: T={args.sweep_temperature:g} is not a pre-registered sweep point "
                  f"{tuple(f'{t:g}' for t in SWEEP_TEMPERATURES)}.")
            return 1
        temperatures = (args.sweep_temperature,)

    summary = {}
    for temperature in temperatures:
        prefix = f"D_T{int(temperature)}"
        result = run_arm("D", len(SWEEP_SEEDS), args,
                         temperature=temperature, label_prefix=prefix)
        if result is None:
            return 1
        summary[prefix] = result
        print(f"{prefix}: {len(result['completed'])} completed")

    # One discards file per process, so two notebooks never write the same Drive file.
    name = ("discards_sweep.json" if args.sweep_temperature is None
            else f"discards_sweep_T{int(args.sweep_temperature)}.json")
    _write_discards(args.runs_dir, summary, name=name)
    return 0


def _write_discards(runs_dir, summary, name="discards.json"):
    """Per-arm discard counts, reported per arm and never aggregated (section 10)."""
    path = pathlib.Path(runs_dir) / name
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {path}")


PHASES = {0: phase0, 1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--phase", default="all",
                        help="'all', or a single phase number 0-5")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip Phase 0 (only for a resumed session that already passed)")
    parser.add_argument("--skip-gate", action="store_true",
                        help="proceed past a failed teacher gate, after investigating it")
    parser.add_argument("--sweep-temperature", type=float, default=None,
                        help="Phase 5 only: run just this pre-registered temperature, so two "
                             "GPU notebooks can split the sweep")
    args = parser.parse_args()

    pathlib.Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    phases = sorted(PHASES) if args.phase == "all" else [int(args.phase)]

    for number in phases:
        code = PHASES[number](args)
        if code:
            print(f"\nstopped in Phase {number} (exit {code})")
            return code

    print("\nall requested phases complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
