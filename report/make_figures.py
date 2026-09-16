"""Every figure in the report, reproducibly, from report/data/ alone. Torch-free.

Architecture diagrams (1-3) need no data. Result figures (4-7) read
report/data/runs/<label>/{metrics.jsonl,test_results.json} and are skipped with a message
when those files are not present yet.

    python report/make_figures.py
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from src.metrics_io import read_metrics  # noqa: E402  (torch-free)

DATA = ROOT / "data" / "runs"
OUT = ROOT / "figures"

# Categorical palette validated with the dataviz validator (light surface): all hard gates
# pass. T, A and D -- the comparison the assignment names -- take slots 1-3, which also
# validate all-pairs. Three hues are below 3:1 contrast, so identity is never colour alone:
# every arm also has its own line style and appears in a legend.
ARM = {
    "T": {"color": "#2a78d6", "ls": "-", "name": "T  teacher ResNet-34 fp32"},
    "A": {"color": "#eb6834", "ls": "--", "name": "A  ResNet-18 fp32, CE"},
    "D": {"color": "#1baf7a", "ls": "-", "name": "D  ResNet-18 ternary, KD"},
    "B": {"color": "#eda100", "ls": "-.", "name": "B  ResNet-18 fp32, KD"},
    "C": {"color": "#e87ba4", "ls": ":", "name": "C  ResNet-18 ternary, CE"},
    "E": {"color": "#008300", "ls": (0, (5, 1, 1, 1)), "name": "E  ResNet-44 fp32, CE"},
}
TERNARY_COLOR, FP32_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5df", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote figures/{name}.png")


# ---------------------------------------------------------------- diagrams

def box(ax, x, y, w, h, text, fc, ec=None, fontsize=8, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, color=color, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-", lw=1.2, rad=0.0, head=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=head,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def fig1_student_architecture():
    """Which layers are ternary and which stay fp32."""
    fig, ax = canvas(7.2, 3.1, (0, 14.4), (-0.4, 6.2))
    tern_fc, fp_fc = "#dbe8f8", "#fbe3d8"

    box(ax, 0.1, 2.3, 1.2, 1.4, "input\n3×32×32", "#f1f0ea", fontsize=7.5)
    box(ax, 1.7, 2.3, 1.5, 1.4, "stem conv\n3×3, 64\nstride 1\nfp32", fp_fc, FP32_COLOR, 7.5)
    arrow(ax, 1.3, 3.0, 1.7, 3.0)

    stages = [("layer1", 64, "32×32", False), ("layer2", 128, "16×16", True),
              ("layer3", 256, "8×8", True), ("layer4", 512, "4×4", True)]
    x = 3.6
    for name, ch, res, ds in stages:
        ax.add_patch(FancyBboxPatch((x - 0.12, 0.45), 2.04, 5.2,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="none", ec=GRID, lw=1.0))
        ax.text(x + 0.9, 5.4, f"{name}\n{ch} ch, {res}", ha="center", va="center", fontsize=7.5,
                weight="bold")
        box(ax, x, 3.25, 1.8, 1.55, "BasicBlock 1\n3×3 conv ×2\nternary", tern_fc,
            TERNARY_COLOR, 7)
        box(ax, x, 1.55, 1.8, 1.55, "BasicBlock 2\n3×3 conv ×2\nternary", tern_fc,
            TERNARY_COLOR, 7)
        if ds:
            box(ax, x + 0.15, 0.6, 1.5, 0.75, "1×1 downsample\nternary", tern_fc,
                TERNARY_COLOR, 6.5)
        arrow(ax, x - 0.4, 3.0, x - 0.12, 3.0)
        x += 2.3
    box(ax, 12.65, 2.35, 1.72, 1.3, "avg pool\n+ FC 512→10\nfp32", fp_fc, FP32_COLOR, 7)
    arrow(ax, x - 0.4, 3.0, 12.65, 3.0)

    ax.text(13.55, 2.1, "BatchNorm after\nevery conv: fp32", fontsize=7, color=MUTED,
            ha="center", va="top")
    for i, (fc, ec, label) in enumerate([(tern_fc, TERNARY_COLOR, "ternary {−α, 0, +α}, per-filter α"),
                                         (fp_fc, FP32_COLOR, "full precision (fp32)")]):
        ax.add_patch(FancyBboxPatch((0.2 + i * 5.2, -0.3), 0.35, 0.3,
                                    boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=1.0))
        ax.text(0.7 + i * 5.2, -0.15, label, va="center", fontsize=7.5)
    ax.text(14.3, -0.15, "19 ternary convs = 99.85% of weights", va="center", ha="right", fontsize=7.5,
            color=MUTED)
    save(fig, "fig1_student_architecture")


def fig2_ternary_block():
    """Latent weights, the quantizer, and the straight-through estimator."""
    fig, ax = canvas(7.2, 3.4, (0, 16), (0, 7))
    latent, quant, conv = "#fbe3d8", "#f1f0ea", "#dbe8f8"

    box(ax, 0.05, 3.6, 3.35, 2.0, "latent weights  W\nfp32, updated by SGD\nclamped to [−1, 1]",
        latent, FP32_COLOR, 7)
    box(ax, 3.9, 3.6, 4.35, 2.0,
        "ternarize, per output filter k\nΔₖ = 0.75 · mean|Wₖ|\nWₜ = sign(W) · 1[|W| > Δₖ]\n"
        "αₖ = mean of |W| where |W| > Δₖ", quant, MUTED, 7)
    box(ax, 8.75, 3.6, 3.15, 2.0, "Wₜ = αₖ · {−1, 0, +1}\nternary weights", conv, TERNARY_COLOR,
        7)
    box(ax, 12.4, 3.6, 3.55, 2.0, "conv(x, Wₜ)\n→ BN → ReLU\nuses ternary\nweights only",
        conv, TERNARY_COLOR, 7)

    for x0, x1 in ((3.4, 3.9), (8.25, 8.75), (11.9, 12.4)):
        arrow(ax, x0, 4.6, x1, 4.6, color=INK, lw=1.4)
    ax.text(8.0, 6.3, "forward pass", ha="center", fontsize=8, weight="bold")

    arrow(ax, 14.15, 3.55, 1.65, 3.55, color=TERNARY_COLOR, ls="--", lw=1.6, rad=-0.28, head=10)
    ax.text(8.0, 0.55,
            "backward: identity straight-through estimator,  ∂L/∂W := ∂L/∂Wₜ\n"
            "the gradient bypasses the quantizer unchanged (no masking, no rescaling)",
            ha="center", va="center", fontsize=7.5, color=INK)
    save(fig, "fig2_ternary_block")


def fig3_kd_setup():
    """Teacher frozen, same augmented batch, the combined loss."""
    fig, ax = canvas(7.2, 3.6, (0, 16), (0, 7.4))
    teacher, student, loss = "#f1f0ea", "#dbe8f8", "#fbe3d8"

    box(ax, 0.05, 2.8, 2.95, 1.7, "augmented\nbatch x\n(crop + flip),\nsame batch to both",
        "#f1f0ea", MUTED, 6.8)
    box(ax, 3.6, 4.8, 4.1, 1.9,
        "teacher: ResNet-34, fp32\nepoch-160 checkpoint\neval() · no_grad() · frozen",
        teacher, MUTED, 7.2)
    box(ax, 3.6, 0.6, 4.1, 1.9, "student: ResNet-18\nternary weights (Fig. 2)\ntrain()",
        student, TERNARY_COLOR, 7.2)
    arrow(ax, 3.0, 4.1, 3.6, 5.6)
    arrow(ax, 3.0, 3.2, 3.6, 1.7)

    box(ax, 8.3, 5.05, 3.0, 1.4, "softmax(zₜ / T)\nsoft targets", teacher, MUTED, 7.2)
    box(ax, 8.3, 2.95, 3.0, 1.4, "log softmax(zₛ / T)", student, TERNARY_COLOR, 7.2)
    box(ax, 8.3, 0.85, 3.0, 1.4, "CE(zₛ, y)\nhard labels", student, TERNARY_COLOR, 7.2)
    arrow(ax, 7.7, 5.75, 8.3, 5.75)
    arrow(ax, 7.7, 2.0, 8.3, 3.5)
    arrow(ax, 7.7, 1.5, 8.3, 1.55)

    box(ax, 12.0, 2.3, 3.9, 3.0,
        "L = (1 − λ) · CE\n+ λ · T² · KL(pₜ ‖ pₛ)\n\nλ = 0.9,  T = 4\nKL: batchmean",
        loss, FP32_COLOR, 7.5)
    arrow(ax, 11.3, 5.6, 12.0, 4.6)
    arrow(ax, 11.3, 3.65, 12.0, 3.8)
    arrow(ax, 11.3, 1.55, 12.0, 2.9)

    arrow(ax, 13.95, 2.25, 5.65, 0.55, color=TERNARY_COLOR, ls="--", lw=1.5, rad=-0.22,
          head=10)
    ax.text(12.6, 0.3, "gradients update the student only", ha="center", fontsize=7.2,
            color=TERNARY_COLOR)
    ax.text(5.65, 7.05, "no gradient reaches the teacher", ha="center", fontsize=7.2,
            color=MUTED)
    save(fig, "fig3_kd_setup")


# ------------------------------------------------------------ result data

def runs_by_group():
    groups = {}
    if not DATA.exists():
        return groups
    for manifest in sorted(DATA.glob("*/manifest.json")):
        if json.loads(manifest.read_text(encoding="utf-8")).get("discard_status") != "completed":
            continue
        label = manifest.parent.name
        groups.setdefault(label.rsplit("_", 1)[0], []).append(manifest.parent)
    return groups


def series(run_dir, key, layer=None):
    xs, ys = [], []
    for record in read_metrics(run_dir / "metrics.jsonl"):
        value = (record.get("per_layer", {}).get(layer, {}).get(key) if layer
                 else record.get(key))
        if value is not None:
            xs.append(record["epoch"])
            ys.append(value)
    return xs, ys


def mean_series(run_dirs, key):
    per_epoch = {}
    for run_dir in run_dirs:
        for x, y in zip(*series(run_dir, key)):
            per_epoch.setdefault(x, []).append(y)
    xs = sorted(per_epoch)
    return xs, [sum(per_epoch[x]) / len(per_epoch[x]) for x in xs]


def lr_drops(ax):
    for epoch in (80, 120):
        ax.axvline(epoch, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=0)


def fig4_learning_curves(groups):
    headline = [g for g in ARM if g in groups]
    if not headline:
        print("skip fig4: no run data in report/data/runs yet")
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    panels = [
        (axes[0, 0], "train_loss_mean", [g for g in ("T", "A", "C", "E") if g in groups],
         "(a) training loss, cross-entropy arms", "mean training loss", True),
        (axes[0, 1], "train_loss_mean", [g for g in ("B", "D") if g in groups],
         "(b) training loss, distillation arms", "mean KD loss (T = 4)", True),
        (axes[1, 0], "val_accuracy", headline,
         "(c) validation accuracy, all arms", "validation accuracy (%)", False),
        (axes[1, 1], "val_accuracy", headline,
         "(d) validation accuracy, epochs 120–160", "validation accuracy (%)", False),
    ]
    for ax, key, arms, title, ylabel, log in panels:
        for arm in arms:
            style = ARM[arm]
            for run_dir in groups[arm]:  # individual seeds, recessive
                xs, ys = series(run_dir, key)
                ax.plot(xs, ys, color=style["color"], lw=0.6, alpha=0.3)
            xs, ys = mean_series(groups[arm], key)
            ax.plot(xs, ys, color=style["color"], ls=style["ls"], lw=1.6,
                    label=f"{style['name']} (n={len(groups[arm])})")
        lr_drops(ax)
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
    axes[1, 1].set_xlim(118, 161)
    lo = min(min(series(r, "val_accuracy")[1][119:] or [100]) for g in headline
             for r in groups[g])
    axes[1, 1].set_ylim(max(lo - 0.3, 80), None)
    axes[1, 0].set_ylim(30, 100)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, -0.1, "Thick line: mean over seeds. Thin lines: individual seeds. "
             "Dotted verticals: learning-rate drops at epochs 80 and 120. "
             "KD and CE losses are on different scales and are plotted separately.",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "fig4_learning_curves")


def depth_colors(n):
    cmap = plt.get_cmap("Blues")
    return [cmap(0.35 + 0.6 * i / max(n - 1, 1)) for i in range(n)]


def layer_names(run_dir):
    for record in read_metrics(run_dir / "metrics.jsonl"):
        if record.get("per_layer"):
            return list(record["per_layer"])
    return []


def per_layer_panel(ax, run_dirs, key, title):
    names = layer_names(run_dirs[0])
    colors = depth_colors(len(names))
    for name, color in zip(names, colors):
        per_epoch = {}
        for run_dir in run_dirs:
            for x, y in zip(*series(run_dir, key, layer=name)):
                per_epoch.setdefault(x, []).append(y)
        xs = sorted(per_epoch)
        ax.plot(xs, [sum(per_epoch[x]) / len(per_epoch[x]) for x in xs], color=color, lw=1.0)
    lr_drops(ax)
    ax.set_title(title, loc="left")
    ax.set_xlabel("epoch")
    return names, colors


def depth_colorbar(fig, axes, names):
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "depth", depth_colors(len(names))), norm=plt.Normalize(1, len(names)))
    bar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("ternary layer (1 = first after stem, 19 = last)", fontsize=7.5)
    bar.set_ticks([1, len(names)])


def fig5_reversal_rate(groups):
    arms = [g for g in ("C", "D") if g in groups]
    if not arms:
        print("skip fig5: no ternary run data yet")
        return
    fig, axes = plt.subplots(1, 1 + len(arms), figsize=(7.2, 2.6), sharey=True)
    ax = axes[0]
    for arm in arms:
        xs, ys = mean_series(groups[arm], "aggregate_reversal_rate")
        ax.plot(xs, ys, color=ARM[arm]["color"], ls=ARM[arm]["ls"], lw=1.6,
                label=ARM[arm]["name"])
    lr_drops(ax)
    ax.set_yscale("log")
    ax.set_title("(a) all ternary weights", loc="left")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reversal rate per epoch")
    ax.legend(fontsize=7, loc="lower left")
    names = []
    for i, arm in enumerate(arms, start=1):
        names, _ = per_layer_panel(axes[i], groups[arm], "reversal_rate",
                                   f"({'bc'[i - 1]}) per layer, arm {arm}")
    depth_colorbar(fig, axes, names)
    save(fig, "fig5_reversal_rate")


def fig6_zero_fraction(groups):
    arms = [g for g in ("C", "D") if g in groups]
    if not arms:
        print("skip fig6: no ternary run data yet")
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(7.2, 2.6), sharey=True, squeeze=False)
    axes = axes[0]
    names = []
    for i, arm in enumerate(arms):
        axes[i].axhspan(0.30, 0.50, color="#f1f0ea", zorder=0)
        axes[i].text(158, 0.305, "TTQ's lowest-error band, 30–50%", fontsize=6.5, color=MUTED,
                     va="bottom", ha="right")
        names, _ = per_layer_panel(axes[i], groups[arm], "zero_fraction",
                                   f"({'ab'[i]}) zero fraction per layer, arm {arm}")
    axes[0].set_ylabel("fraction of weights quantized to 0")
    depth_colorbar(fig, list(axes), names)
    save(fig, "fig6_zero_fraction")


def fig7_temperature_sweep(groups):
    points = {}
    for temperature, group in ((2, "D_T2"), (4, "D"), (8, "D_T8")):
        values = []
        for run_dir in groups.get(group, []):
            seed = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["seed"]
            results = run_dir / "test_results.json"
            if seed in (0, 1) and results.exists():  # T=4 uses seeds 0 and 1 only (section 9)
                values.append(json.loads(results.read_text(encoding="utf-8"))["last10_mean"])
        if len(values) == 2:
            points[temperature] = values
    if len(points) < 3:
        print(f"skip fig7: {len(points)} of 3 temperature points have test results")
        return

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    c_values = [json.loads((r / "test_results.json").read_text(encoding="utf-8"))["last10_mean"]
                for r in groups.get("C", []) if (r / "test_results.json").exists()]
    if c_values:
        c_mean = sum(c_values) / len(c_values)
        ax.axhspan(min(c_values), max(c_values), color=ARM["C"]["color"], alpha=0.15, lw=0)
        ax.axhline(c_mean, color=ARM["C"]["color"], ls=ARM["C"]["ls"], lw=1.4)
        ax.text(8.25, c_mean, f"C, no teacher\n(n={len(c_values)}, band = range)",
                fontsize=7, va="center")
    xs = sorted(points)
    means = [sum(points[t]) / 2 for t in xs]
    for t in xs:
        ax.plot([t, t], [min(points[t]), max(points[t])], color=ARM["D"]["color"], lw=2)
        ax.scatter([t, t], points[t], s=22, color=ARM["D"]["color"], ec=SURFACE, lw=1.2,
                   zorder=3)
    ax.plot(xs, means, color=ARM["D"]["color"], lw=1.6, marker="o", ms=4, label="D mean")
    for t, m in zip(xs, means):
        ax.annotate(f"{m:.2f}", (t, m), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(t) for t in xs])
    ax.set_xlabel("distillation temperature T")
    ax.set_ylabel("test accuracy, last-10 mean (%)")
    ax.set_title("Arm D across temperatures (2 seeds per point)", loc="left")
    save(fig, "fig7_temperature_sweep")


def main():
    fig1_student_architecture()
    fig2_ternary_block()
    fig3_kd_setup()
    groups = runs_by_group()
    fig4_learning_curves(groups)
    fig5_reversal_rate(groups)
    fig6_zero_fraction(groups)
    fig7_temperature_sweep(groups)


if __name__ == "__main__":
    main()
