"""Regenerate notebooks/colab_runner.ipynb with one cell per phase."""

import json
import pathlib

REPO = pathlib.Path(r"D:\ATDL Assignment 1")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


cells = [
    md("""# Ternary Weight Quantization + KD on CIFAR-10

Thin launcher. All the logic lives in `src/` and `scripts/` in the repo; these cells only
mount Drive, pull the code, and call it.

**Run the cells in order.** Each numbered section below is one cell to run. Sections 1-3
are setup and take under a minute. Sections 4 onward are the study itself.

**When Colab disconnects** - over 20-35 GPU-hours it will - reopen this notebook, re-run
sections 1, 2 and 3, then re-run whichever training section you were in. Finished runs are
skipped and a partial run resumes from its last epoch, so re-running is always safe and
never repeats work."""),

    md("## 1. Environment\n\nConfirms you have a GPU and prints which one."),
    code("""import torch

assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4/L4/A100 GPU."
print("gpu        ", torch.cuda.get_device_name(0))
print("torch      ", torch.__version__, "| cuda", torch.version.cuda)
print("tf32 matmul", torch.backends.cuda.matmul.allow_tf32,
      "| tf32 cudnn", torch.backends.cudnn.allow_tf32)
print()
print("The pre-registration is fp32-only and forbids TF32. src/train.py forces both")
print("flags False at run start, so the values above are just Colab defaults.")"""),

    md("""## 2. Drive

`runs/` must live on Drive - it is the only thing that survives a disconnect. CIFAR-10
stays on local disk, where it re-downloads in seconds and reads much faster."""),
    code("""import os
from google.colab import drive

drive.mount('/content/drive')

RUNS_DIR = '/content/drive/MyDrive/atdl_runs'   # persistent
DATA_DIR = '/content/data'                      # ephemeral, re-downloaded per session
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(f"{RUNS_DIR}/logs", exist_ok=True)
print(RUNS_DIR, '->', sorted(os.listdir(RUNS_DIR)))"""),

    md("""## 3. Code

Pulls the latest commit. Every run manifest records the commit it trained under, which is
why the code comes from git rather than from pasted cells."""),
    code("""REPO_DIR = "/content/atdl"
REPO_URL = "https://github.com/pbhatt0022/ternary-kd-cifar10.git"

import os, subprocess
if os.path.isdir(f"{REPO_DIR}/.git"):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)

subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("commit", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True).stdout.strip())"""),

    md("""## 4. Phase 0 - pre-flight

Unit tests, then the hardware checks: determinism, the LR schedule boundaries, the
ternarized layer set, the weight-decay split, and data hygiene. Trains six short epochs to
prove two runs at the same seed agree exactly.

**Takes a few minutes.** Nothing trains for real until this passes. It records its pass
against the current commit, so re-running it later is instant unless the code changed."""),
    code("""!python -u scripts/run_all.py --phase 0 --runs-dir {RUNS_DIR} --data-dir {DATA_DIR} 2>&1 | tee -a {RUNS_DIR}/logs/phase0.log"""),

    md("""## 5. Phases 1 and 2 - arm E check, then the teacher

Phase 1 confirms the committed arm E architecture still matches what the code computes.
Phase 2 trains the ResNet-34 teacher, checks it against the 92% validation gate, and
reloads a checkpoint to confirm it reproduces its logged accuracy exactly.

**This is the first real training run: roughly 1-3 hours.** The first epoch prints a
measured seconds-per-epoch figure and a projection, so you will know the true budget about
two minutes in. If that projection looks wrong, stop and say so before continuing."""),
    code("""!python -u scripts/run_all.py --phase 1 --runs-dir {RUNS_DIR} --data-dir {DATA_DIR} 2>&1 | tee -a {RUNS_DIR}/logs/phase1.log
!python -u scripts/run_all.py --phase 2 --runs-dir {RUNS_DIR} --data-dir {DATA_DIR} 2>&1 | tee -a {RUNS_DIR}/logs/phase2.log"""),

    md("""## 6. Phase 3 - the headline arms

A (3 seeds), B (2), C (3), D (3), E (1): twelve runs, sequentially. **This is the bulk of
the compute - budget most of the total here.**

Expect to re-run this cell several times across sessions. Completed runs are skipped
instantly and a partial run picks up from its last finished epoch."""),
    code("""!python -u scripts/run_all.py --phase 3 --runs-dir {RUNS_DIR} --data-dir {DATA_DIR} 2>&1 | tee -a {RUNS_DIR}/logs/phase3.log"""),

    md("""## 7. Phase 4 - evaluation

The first and only time the test set is touched. Loads the ten checkpoints from epochs
151-160 for every arm and writes `test_results.json`. Minutes, not hours."""),
    code("""!python -u scripts/run_all.py --phase 4 --runs-dir {RUNS_DIR} --data-dir {DATA_DIR} 2>&1 | tee -a {RUNS_DIR}/logs/phase4.log"""),

    md("""## 8. Phase 5 - temperature sweep

Arm D at T=2 and T=8, seeds 0 and 1: four more runs. Refuses to start until every headline
arm has its Phase 4 numbers, because the sweep must not be able to revise them."""),
    code("""!python -u scripts/run_all.py --phase 5 --runs-dir {RUNS_DIR} --data-dir {DATA_DIR} 2>&1 | tee -a {RUNS_DIR}/logs/phase5.log"""),

    md("""## 9. Tables

Reads only the files under `runs/`. No reported number is computed by hand."""),
    code("""!python -u scripts/aggregate.py --runs-dir {RUNS_DIR} --out {RUNS_DIR}/report 2>&1 | tee -a {RUNS_DIR}/logs/aggregate.log"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}
for cell in nb["cells"]:
    cell["source"] = cell["source"].splitlines(keepends=True)

out = REPO / "notebooks" / "colab_runner.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

import ast
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if src.lstrip().startswith("!"):
        print(f"cell {i}: shell magic, skipped")
        continue
    ast.parse(src)
    print(f"cell {i}: compiles")
print(f"\nwrote {out} with {len(cells)} cells")
