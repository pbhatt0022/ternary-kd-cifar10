# Ternary-weight ResNet-18 with knowledge distillation on CIFAR-10

A ResNet-34 full-precision teacher distils into a ResNet-18 student whose convolutional weights
are constrained to {−α, 0, +α} throughout training (quantization-aware training with a
straight-through estimator). The study is pre-registered: every design choice and every rule for
interpreting the results was fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md) before training,
and deviations are recorded in its append-only amendment log.

## Where each assignment deliverable lives

| Deliverable | File |
|---|---|
| Data pipeline | [`src/data.py`](src/data.py) — CIFAR-10, fixed 45k/5k train/validation split, augmentation |
| Teacher training | [`src/train.py`](src/train.py), run as arm `T` by [`scripts/run_all.py`](scripts/run_all.py) (Phase 2) |
| Ternary layer | [`src/quantizer.py`](src/quantizer.py) (TWN per-filter ternarization), [`src/ternary_modules.py`](src/ternary_modules.py) (`TernaryConv2d`, identity STE) |
| KD + QAT training | [`src/losses.py`](src/losses.py) (KD loss), [`src/train.py`](src/train.py), run as arm `D` (Phase 3) |
| Evaluation | [`src/evaluate.py`](src/evaluate.py), [`scripts/evaluate_all.py`](scripts/evaluate_all.py) (Phase 4), [`scripts/aggregate.py`](scripts/aggregate.py) (tables) |
| Ternary constraint check on saved weights | [`scripts/verify_ternary.py`](scripts/verify_ternary.py) |
| Fixed seeds | Seeds per run in `scripts/run_all.py`; split seed `20260828` in `src/data.py`; set in `src/train.py:set_determinism` |
| Environment | [`requirements.txt`](requirements.txt) |

## Experimental arms

| Arm | Model | Weights | Loss | Seeds |
|---|---|---|---|---|
| T | ResNet-34 | fp32 | cross-entropy | 1 |
| A | ResNet-18 | fp32 | cross-entropy | 3 |
| B | ResNet-18 | fp32 | cross-entropy + KD from T | 2 |
| C | ResNet-18 | ternary | cross-entropy | 3 |
| D | ResNet-18 | ternary | cross-entropy + KD from T | 3 |
| E | ResNet-44 | fp32 | cross-entropy (nearest-storage control) | 1 |
| Sweep | as D, T ∈ {2, 8} | ternary | cross-entropy + KD | 2 per point |

## Running it

Training was run on Google Colab (T4 GPU) through
[`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb), which mounts Google Drive, clones
this repository, and calls the scripts below. Outputs go to Drive so they survive disconnects.

```
pip install -r requirements.txt
python scripts/run_all.py --phase 0 --runs-dir RUNS --data-dir DATA   # tests + pre-flight checks
python scripts/run_all.py --phase 1 --runs-dir RUNS --data-dir DATA   # confirm arm E architecture
python scripts/run_all.py --phase 2 --runs-dir RUNS --data-dir DATA   # teacher, 92% gate, restore check
python scripts/run_all.py --phase 3 --runs-dir RUNS --data-dir DATA   # arms A, B, C, D, E
python scripts/run_all.py --phase 4 --runs-dir RUNS --data-dir DATA   # the single test-set evaluation
python scripts/run_all.py --phase 5 --runs-dir RUNS --data-dir DATA   # temperature sweep
python scripts/aggregate.py --runs-dir RUNS --out RUNS/report         # tables and curve data
```

Every phase is idempotent: finished runs are skipped and an interrupted run resumes from its last
completed epoch. Phase 5 can be split across notebooks with `--sweep-temperature` and
`--sweep-seed`; a heartbeat lock stops two machines from training the same run.

### Verifying the ternary constraint on a saved checkpoint

```
python scripts/verify_ternary.py --checkpoint RUNS/D_0/checkpoints/epoch_160.pt \
    --export student_ternary_2bit.pth
```

This loads the saved latent weights, ternarizes every quantized layer, checks that each output
filter holds only values in {−α, 0, +α}, confirms which layers remain fp32, and reports per-layer
sparsity. With `--export` it also writes the model in its deployed form, packing ternary weights
at 2 bits each, then reloads that file and checks it reproduces the ternary weights exactly.

## Run data

All training and evaluation outputs are on Google Drive (view only):
[`atdl_runs`](https://drive.google.com/drive/folders/1lmWoubSkr0ktE2DyV8at0H1bcoFZnI1r?usp=drive_link).
Each run folder holds `manifest.json` (seed, commit, GPU, hyperparameters, status),
`metrics.jsonl` (one record per epoch), `test_results.json`, and the checkpoints from epochs
151–160. `report/` holds the tables and curve data from `scripts/aggregate.py`, and `logs/` the
console output of every phase.

## Checkpoints

Download (view only):
[`atdl_submission`](https://drive.google.com/drive/folders/17othiYp228EzTOTxicTBcAypWRbcZFXd?usp=sharing),
containing `teacher_resnet34_epoch160.pth`, `student_ternary_resnet18_epoch160.pth` (fp32 latent
weights) and `student_ternary_resnet18_2bit.pth` (the deployable 2-bit export).

The study deliberately performs **no best-checkpoint selection**: choosing the best epoch biases
results upward in proportion to a run's variance. The submitted checkpoints are therefore the
**final (epoch 160)** checkpoints of the teacher (`T_0`) and of the seed-0 ternary KD student
(`D_0`), not a "best" epoch. The reported metric is the mean test accuracy over the ten
checkpoints from epochs 151–160.

Checkpoints store fp32 **latent** weights, which is what training updates. The ternary model is
recomputed from them deterministically, which is what `verify_ternary.py` demonstrates.

## Tests

```
python -m pytest tests -q
```

59 tests cover the quantizer, the straight-through estimator, the KD loss, the reversal-rate
metric, storage accounting (against an independent torch-free recomputation), the section 8
interpretation rules, metrics reading, and the run lock. `scripts/verify.py` adds hardware checks:
determinism, LR schedule boundaries, the ternarized layer set, weight-decay grouping, data hygiene,
checkpoint restore, and that a resumed run reproduces an uninterrupted one exactly.

## Reproducibility notes

Runs are deterministic given a seed on a fixed GPU model (`cudnn.deterministic = True`, TF32
disabled, fp32 throughout). Execution deviations found during the study, including one run whose
data order does not match its seed after a resume, are documented in amendment A5 of the
pre-registration.
