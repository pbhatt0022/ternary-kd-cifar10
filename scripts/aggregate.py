"""Produce every reported number from the files under runs/. Pre-registration sections 7-8.

Nothing in the writeup is computed by hand. The section 8 interpretation rules are
implemented as code here and emit their own verdicts, so the reading of the result is fixed
in advance rather than chosen after seeing it.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import storage
from src.metrics_io import read_metrics
from src.train import ARM_SPEC, CHECKPOINT_EPOCHS, build_model

HEADLINE_ORDER = ("T", "A", "B", "C", "D", "E")
SEEDS_EXPECTED = {"T": 1, "A": 3, "B": 2, "C": 3, "D": 3, "E": 1}
SUGGESTIVE_MULTIPLIER = 2
SWEEP_SEEDS = (0, 1)  # section 9: every sweep point rests on seeds 0 and 1


def spread(values):
    """max - min, in percentage points. The only dispersion measure this study uses."""
    return max(values) - min(values) if len(values) > 1 else None


def load_runs(runs_dir):
    """All test_results.json, plus the discard verdicts, grouped by arm label prefix."""
    runs_dir = pathlib.Path(runs_dir)
    results, discarded = {}, {}

    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        label = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = manifest.get("discard_status", "running")
        group = label.rsplit("_", 1)[0]  # "D_0" -> "D", "D_T2_0" -> "D_T2"

        if status.startswith("discarded"):
            discarded.setdefault(group, []).append((label, status))
            continue
        if status != "completed":
            continue

        results_path = manifest_path.parent / "test_results.json"
        if results_path.exists():
            record = json.loads(results_path.read_text(encoding="utf-8"))
            record["metrics_path"] = manifest_path.parent / "metrics.jsonl"
            record["manifest"] = manifest
            results.setdefault(group, []).append(record)

    return results, discarded


def model_facts(arm):
    """Architecture-only figures: parameters, bits/weight, exported storage, MACs."""
    model, ternary_names = build_model(arm, "cpu")
    if ternary_names:
        bits = storage.ternary_storage_bits(model, ternary_names)
    else:
        bits = storage.fp32_storage_bits(model)
    return {
        "parameters": storage.n_parameters(model),
        "bits_per_weight": storage.bits_per_weight(model, ternary_names),
        "storage_mb": storage.mb(bits),
        "macs": storage.macs(model),
    }


def arm_summary(group, records, discarded):
    means = [r["last10_mean"] for r in records]
    arm = records[0]["arm"]
    facts = model_facts(arm)
    return {
        "group": group,
        "arm": arm,
        "seeds": sorted(r["seed"] for r in records),
        "n_seeds": len(records),
        "last10_test_mean": sum(means) / len(means),
        "last10_test_spread": spread(means),
        "epoch160_test_mean": sum(r["epoch160_accuracy"] for r in records) / len(records),
        "unsettled": [r["label"] for r in records if r["settledness_flag"]],
        "discards": len(discarded.get(group, [])),
        **facts,
    }


def interpret(summaries):
    """The section 8 rules, verbatim, as code. Each returns its own verdict string."""
    verdicts = {}
    if not all(key in summaries for key in "ABCD"):
        verdicts["status"] = "incomplete: need arms A, B, C and D for the section 8 rules"
        return verdicts

    a, b, c, d = (summaries[k]["last10_test_mean"] for k in "ABCD")
    spreads = [summaries[k]["last10_test_spread"] for k in "ABCD"]
    spreads = [s for s in spreads if s is not None]
    max_spread = max(spreads) if spreads else None
    verdicts["max_within_arm_spread"] = max_spread
    verdicts["max_spread_note"] = (
        "includes arm B, whose spread comes from 2 seeds rather than 3"
    )

    # Recovery ratio
    if max_spread is None:
        verdicts["recovery_ratio"] = None
        verdicts["recovery_verdict"] = "undefined; no multi-seed spread available"
    elif c >= a or abs(a - c) < max_spread:
        verdicts["recovery_ratio"] = None
        verdicts["recovery_verdict"] = (
            "undefined; report raw accuracies and use no 'recovery' language. "
            + ("C >= A: ternarization matched or exceeded full precision"
               if c >= a else
               f"|A - C| = {abs(a - c):.2f}pp is smaller than the largest within-arm "
               f"spread ({max_spread:.2f}pp)")
        )
    else:
        verdicts["recovery_ratio"] = (d - c) / (a - c)
        verdicts["recovery_verdict"] = "defined"

    # Teacher comparison, which can invalidate the interaction term entirely
    teacher_ok = True
    if "T" in summaries:
        t = summaries["T"]["last10_test_mean"]
        verdicts["teacher_test"] = t
        if t <= a:
            teacher_ok = False
            verdicts["teacher_verdict"] = (
                f"T ({t:.2f}%) is at or below arm A's mean ({a:.2f}%): the teacher provides "
                "no capability the student lacks. Arms B and D are uninterpretable as "
                "distillation results and the interaction term is NOT computed."
            )
        else:
            verdicts["teacher_verdict"] = (
                f"T ({t:.2f}%) exceeds arm A's mean ({a:.2f}%); B and D are interpretable "
                "as distillation."
            )

    # Interaction
    if not teacher_ok:
        verdicts["interaction"] = None
        verdicts["interaction_verdict"] = "not computed; see the teacher comparison"
    else:
        interaction = (d - c) - (b - a)
        verdicts["interaction"] = interaction
        if max_spread is not None and abs(interaction) > SUGGESTIVE_MULTIPLIER * max_spread:
            verdicts["interaction_verdict"] = (
                "suggestive (single-seed-limited, not an effect size); "
                f"|{interaction:.2f}| exceeds {SUGGESTIVE_MULTIPLIER} x {max_spread:.2f}pp"
            )
        else:
            verdicts["interaction_verdict"] = "unresolved"

    return verdicts


def per_layer_table(records):
    """Zero fraction and degenerate counts per ternarized layer, over epochs 151-160."""
    window = set(CHECKPOINT_EPOCHS)
    layers = {}
    for record in records:
        for entry in read_metrics(record["metrics_path"]):
            for name, values in entry.get("per_layer", {}).items():
                store = layers.setdefault(
                    name, {"zero": [], "degen_window": [], "degen_all": [],
                           "n_weights": values["n_weights"]}
                )
                if values.get("degenerate_filter_count") is not None:
                    store["degen_all"].append(values["degenerate_filter_count"])
                if entry["epoch"] in window:
                    store["zero"].append(values["zero_fraction"])
                    if values.get("degenerate_filter_count") is not None:
                        store["degen_window"].append(values["degenerate_filter_count"])

    rows = []
    for name, store in layers.items():
        rows.append({
            "layer": name,
            "n_weights": store["n_weights"],
            "zero_fraction_last10_mean": (
                sum(store["zero"]) / len(store["zero"]) if store["zero"] else None
            ),
            "degenerate_last10_mean": (
                sum(store["degen_window"]) / len(store["degen_window"])
                if store["degen_window"] else None
            ),
            "degenerate_max_over_training": max(store["degen_all"]) if store["degen_all"] else None,
        })
    return rows


def curves(records, key):
    """Per-epoch series for each seed: [(label, [(epoch, value), ...]), ...]."""
    series = []
    for record in records:
        points = []
        for entry in read_metrics(record["metrics_path"]):
            value = entry.get(key)
            if value is not None:  # reversal rate is null for epochs 1 and 2
                points.append((entry["epoch"], value))
        series.append((record["label"], points))
    return series


def sweep_summary(results):
    """The temperature sweep, per section 9.

    The T=4 point is arm D at seeds 0 and 1 ONLY -- not arm D's three-seed mean. Section 9
    defines it that way, and mixing a 3-seed mean against two 2-seed points would bias the
    comparison. Arm C is the no-teacher reference line.
    """
    points = {}
    for temperature, group in ((2.0, "D_T2"), (4.0, "D"), (8.0, "D_T8")):
        records = [r for r in results.get(group, []) if r["seed"] in SWEEP_SEEDS]
        if len(records) < len(SWEEP_SEEDS):
            continue
        means = [r["last10_mean"] for r in records]
        points[temperature] = {
            "mean": sum(means) / len(means),
            "spread": spread(means),
            "seeds": sorted(r["seed"] for r in records),
            "source": group,
        }

    if len(points) < 3:
        return points, f"incomplete: {len(points)} of 3 temperature points available"

    ordered = [points[t]["mean"] for t in sorted(points)]
    spreads = [points[t]["spread"] for t in points if points[t]["spread"] is not None]
    max_spread = max(spreads) if spreads else None

    rising = all(b > a for a, b in zip(ordered, ordered[1:]))
    falling = all(b < a for a, b in zip(ordered, ordered[1:]))
    difference = max(ordered) - min(ordered)

    if rising or falling:
        verdict = (
            f"monotone {'increasing' if rising else 'decreasing'} in T "
            f"({' -> '.join(f'{m:.2f}%' for m in ordered)})"
        )
    elif max_spread is not None and difference > max_spread:
        verdict = (
            f"non-monotone, but the {difference:.2f}pp range exceeds the largest "
            f"across-seed spread ({max_spread:.2f}pp), so the differences are reported"
        )
    else:
        verdict = (
            f"flat: the {difference:.2f}pp range does not exceed the largest across-seed "
            f"spread ({max_spread:.2f}pp)" if max_spread is not None else "flat"
        )
    return points, verdict


def per_layer_curves(records, field):
    """(label, layer, epoch, value) rows for a per-layer field.

    Null values are skipped rather than zero-filled: the reversal rate is genuinely
    undefined for epochs 1 and 2, and a 0.0 there would be indistinguishable from a stable
    epoch.
    """
    rows = []
    for record in records:
        for entry in read_metrics(record["metrics_path"]):
            for name, values in entry.get("per_layer", {}).items():
                value = values.get(field)
                if value is not None:
                    rows.append((record["label"], name, entry["epoch"], value))
    return rows


def _fmt(value, digits=2, suffix=""):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def write_report(out_dir, summaries, verdicts, results, discarded):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Results", "", "## Headline table", ""]

    header = ("| arm | seeds | last-10 test mean | spread | epoch-160 test | unsettled | "
              "discards | params | bits/weight | storage MB | MACs |")
    lines += [header, "|" + "---|" * 11]

    for group in HEADLINE_ORDER:
        if group not in summaries:
            continue
        s = summaries[group]
        note = " (2 seeds)" if s["n_seeds"] == 2 else (" (1 seed)" if s["n_seeds"] == 1 else "")
        lines.append(
            f"| {s['group']} | {s['n_seeds']} | {s['last10_test_mean']:.2f}% | "
            f"{_fmt(s['last10_test_spread'], suffix='pp')}{note} | "
            f"{s['epoch160_test_mean']:.2f}% | {len(s['unsettled'])} | {s['discards']} | "
            f"{s['parameters']:,} | {s['bits_per_weight']:.2f} | {s['storage_mb']:.4f} | "
            f"{s['macs']:,} |"
        )

    sweep_points, sweep_verdict = sweep_summary(results)
    if sweep_points:
        lines += ["", "## Temperature sweep (section 9)", "",
                  "| T | seeds | source runs | last-10 test mean | spread |",
                  "|" + "---|" * 5]
        for temperature in sorted(sweep_points):
            point = sweep_points[temperature]
            lines.append(
                f"| {temperature:g} | {len(point['seeds'])} | `{point['source']}` | "
                f"{point['mean']:.2f}% | {_fmt(point['spread'], suffix='pp')} (2 seeds) |"
            )
        if "C" in summaries:
            lines.append(
                f"| - | {summaries['C']['n_seeds']} | `C` (no-teacher reference) | "
                f"{summaries['C']['last10_test_mean']:.2f}% | "
                f"{_fmt(summaries['C']['last10_test_spread'], suffix='pp')} |"
            )
        lines += [
            "",
            f"**Sweep verdict:** {sweep_verdict}",
            "",
            "The T=4 point is arm D at seeds 0 and 1 only, per section 9 -- not arm D's",
            "three-seed mean, so all three points rest on the same seed count. The sweep",
            "cannot revise any headline number; a temperature beating T=4 is reported as an",
            "unexploited finding.",
        ]

    lines += ["", "## Interpretation (section 8 rules, applied as code)", ""]
    if "max_within_arm_spread" in verdicts:
        lines.append(
            f"- **Largest within-arm spread:** "
            f"{_fmt(verdicts['max_within_arm_spread'], suffix='pp')} "
            f"({verdicts['max_spread_note']})"
        )
    for key, label in (("teacher_verdict", "Teacher comparison"),
                       ("recovery_verdict", "Recovery ratio"),
                       ("interaction_verdict", "Interaction")):
        if key in verdicts:
            lines.append(f"- **{label}:** {verdicts[key]}")
    if verdicts.get("recovery_ratio") is not None:
        lines.append(f"- R = (D - C) / (A - C) = {verdicts['recovery_ratio']:.3f}")
    if verdicts.get("interaction") is not None:
        lines.append(f"- (D - C) - (B - A) = {verdicts['interaction']:.2f}pp")

    lines += [
        "",
        "Any difference smaller than the largest relevant within-arm spread is reported as",
        "**not resolved**, never as a small effect. Single-seed arms support threshold",
        "judgements and directional statements only.",
        "",
        "## Compression framing",
        "",
    ]
    if "C" in summaries and "A" in summaries:
        lines += [
            f"Ternary ResNet-18 exports at {summaries['C']['storage_mb']:.4f} MB against "
            f"{summaries['A']['storage_mb']:.4f} MB for the fp32 model "
            f"({summaries['A']['storage_mb'] / summaries['C']['storage_mb']:.1f}x smaller), "
            f"at {summaries['C']['bits_per_weight']:.2f} bits per weight.",
            "",
            f"**MACs are unchanged by weight ternarization** "
            f"({summaries['C']['macs']:,} for both), and no latency claim follows without "
            "custom kernels, which were not written. Checkpoints hold fp32 latent weights "
            "and are full size; compression describes the exported model only.",
        ]

    if discarded:
        lines += ["", "## Discards (per arm, never aggregated)", ""]
        for group, items in sorted(discarded.items()):
            for label, status in items:
                lines.append(f"- `{label}`: {status}")
    else:
        lines += ["", "## Discards", "", "None."]

    ternary_groups = [g for g in summaries if ARM_SPEC[summaries[g]["arm"]]["ternary"]]
    for group in sorted(ternary_groups):
        rows = per_layer_table(results[group])
        if not rows:
            continue
        lines += ["", f"## Per-layer ternary metrics: arm {group}", "",
                  "| layer | weights | zero fraction (last-10 mean) | degenerate "
                  "(last-10 mean) | degenerate (max over training) |",
                  "|" + "---|" * 5]
        for row in rows:
            lines.append(
                f"| `{row['layer']}` | {row['n_weights']:,} | "
                f"{_fmt(row['zero_fraction_last10_mean'], 4)} | "
                f"{_fmt(row['degenerate_last10_mean'], 2)} | "
                f"{row['degenerate_max_over_training']} |"
            )
        if all((row["degenerate_max_over_training"] or 0) == 0 for row in rows):
            lines += [
                "",
                "Every degenerate-filter count is zero. This is the expected outcome, not a",
                "logging failure: Delta = 0.75 x mean|w| and max|w| >= mean|w|, so the",
                "support is non-empty for any filter that is not bit-exactly zero, and the",
                "fallback can only fire on the all-zero absorbing state it exists to rescue.",
            ]

    report = out_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_curve_data(out_dir, results):
    """Validation and reversal-rate curves as CSV, one file per series type."""
    written = []
    for key, name in (("val_accuracy", "val_curves.csv"),
                      ("aggregate_reversal_rate", "reversal_curves.csv")):
        rows = ["label,epoch,value"]
        for group, records in sorted(results.items()):
            for label, points in curves(records, key):
                rows += [f"{label},{epoch},{value}" for epoch, value in points]
        if len(rows) > 1:
            path = out_dir / name
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            written.append(path)

    # Per-layer series: reversal rate (section 7 wants these alongside the aggregate) and
    # zero fraction per epoch, which the sparsity-evolution figure needs.
    for field, name in (("reversal_rate", "reversal_curves_per_layer.csv"),
                        ("zero_fraction", "zero_fraction_curves.csv"),
                        ("degenerate_filter_count", "degenerate_curves.csv")):
        rows = [f"label,layer,epoch,{field}"]
        for group, records in sorted(results.items()):
            for label, layer, epoch, value in per_layer_curves(records, field):
                rows.append(f"{label},{layer},{epoch},{value}")
        if len(rows) > 1:
            path = out_dir / name
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            written.append(path)

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    results, discarded = load_runs(args.runs_dir)
    if not results:
        print("no evaluated runs found; Phase 4 must run first")
        return 1

    summaries = {g: arm_summary(g, r, discarded) for g, r in results.items()}
    verdicts = interpret(summaries)

    out_dir = pathlib.Path(args.out)
    report = write_report(out_dir, summaries, verdicts, results, discarded)
    curve_files = write_curve_data(out_dir, results)

    (out_dir / "summaries.json").write_text(
        json.dumps({"summaries": summaries, "verdicts": verdicts}, indent=2, default=str),
        encoding="utf-8",
    )

    print(report.read_text(encoding="utf-8"))
    print(f"\nwrote {report}")
    for path in curve_files:
        print(f"wrote {path}")
    print(f"wrote {out_dir / 'summaries.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
