"""Orchestration. Pre-registration sections 5, 6 and 10.

The only module that knows the phase ordering. Idempotent by design: it decides what to do
by reading runs/ rather than by remembering anything, so re-running it after a Colab
disconnect resumes exactly where it stopped (amendment A3).

    python scripts/run_all.py --runs-dir <dir> --data-dir <dir>
    python scripts/run_all.py --runs-dir <dir> --data-dir <dir> --phase 3
"""

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import train

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
    records = {
        json.loads(l)["epoch"]: json.loads(l)["val_accuracy"]
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    return [records[e] for e in epochs if e in records]


def _run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=REPO).returncode


def phase0(args):
    """Pre-flight. Nothing trains until this passes.

    The determinism check trains six short epochs, so the pass is recorded against the
    commit it passed at: a resumed session at the same commit skips it, and any code
    change forces it to run again.
    """
    print("\n=== Phase 0: pre-flight ===")
    if args.skip_verify:
        print("skipped by --skip-verify")
        return 0

    marker = pathlib.Path(args.runs_dir) / ".preflight_passed"
    commit = train._git_commit()
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == commit:
        print(f"already passed at commit {commit[:8]}")
        return 0

    if _run([sys.executable, "-m", "pytest", "tests", "-q"]):
        print("unit tests FAILED")
        return 1
    code = _run([
        sys.executable, "scripts/verify.py",
        "--data-dir", args.data_dir, "--runs-dir", args.runs_dir,
    ])
    if code == 0 and commit != "unknown":
        marker.write_text(commit, encoding="utf-8")
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

    summary = {}
    for temperature in SWEEP_TEMPERATURES:
        prefix = f"D_T{int(temperature)}"
        result = run_arm("D", len(SWEEP_SEEDS), args,
                         temperature=temperature, label_prefix=prefix)
        if result is None:
            return 1
        summary[prefix] = result
        print(f"{prefix}: {len(result['completed'])} completed")
    _write_discards(args.runs_dir, summary, name="discards_sweep.json")
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
