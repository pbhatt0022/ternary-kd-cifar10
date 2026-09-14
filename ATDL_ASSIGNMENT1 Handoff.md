# Implementation Handoff: Ternary Weight Quantization + KD on CIFAR-10

**Audience:** an agentic coding assistant (Claude Code / Codex) implementing a pre-registered experiment.

**Read this first.** This project is governed by a frozen pre-registration (`PREREGISTRATION.md`, dated and committed before any training). Your job is to implement that document exactly. It is not a starting point to improve on.

The single most important instruction: **when you notice something that looks suboptimal, do not fix it.** Many of the choices here are deliberately suboptimal in ways that were argued about at length and recorded as limitations. Silently improving them destroys the study. If you believe something is wrong, stop and report it rather than changing it. See §10 for the explicit list of things you will be tempted to add and must not.

---

## 1. What is being measured

Five arms on CIFAR-10, disentangling the effect of ternary weight quantization from the effect of knowledge distillation:

| Arm | Model | Weights | Loss | Seeds |
|---|---|---|---|---|
| T | ResNet-34 | fp32 | CE | 1 |
| A | ResNet-18 | fp32 | CE | 3 |
| B | ResNet-18 | fp32 | CE + KD from T | 2 |
| C | ResNet-18 | ternary | CE | 3 |
| D | ResNet-18 | ternary | CE + KD from T | 3 |
| E | ResNet-{20,32,44,56,110} | fp32 | CE | 1 |

Plus a temperature sweep on D at T ∈ {2, 8}, 2 seeds each, run **only after** all the above are complete.

Total: 13 headline runs + 4 sweep runs = **17 runs**. (T=4 at seeds 0 and 1 are already covered by the D arm, so the sweep needs only the two new temperatures.)

---

## 2. Repository layout

```
.
├── PREREGISTRATION.md          # frozen; never edit
├── HANDOFF.md                  # this file
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py                 # CIFAR-10 loading, 45k/5k split, augmentation
│   ├── models.py               # ResNet-18/34 CIFAR-adapted, 6n+2 family
│   ├── quantizer.py            # TWN per-filter ternarization + STE
│   ├── ternary_modules.py      # TernaryConv2d wrapper
│   ├── losses.py               # KD loss
│   ├── tracking.py             # reversal rate, zero fraction, degenerate counts
│   ├── storage.py              # exported-storage accounting, E selection
│   ├── train.py                # single-run training loop
│   └── evaluate.py             # end-of-arm test evaluation
├── scripts/
│   ├── select_arm_e.py         # run once, before training
│   ├── run_all.py              # orchestration
│   ├── verify.py               # pre-flight checks (§9)
│   └── aggregate.py            # produce final tables from runs/
├── tests/
│   └── test_*.py               # see §9
└── runs/                       # created at runtime; see §8
```

---

## 3. Implementation order

Do not train anything until steps 1–4 pass.

1. `src/data.py`, `src/models.py`, `src/storage.py`
2. `src/quantizer.py`, `src/ternary_modules.py`, `src/losses.py`, `src/tracking.py` **with unit tests** (§9)
3. `scripts/select_arm_e.py` — run it, commit the result to the manifest template
4. `scripts/verify.py` — determinism and restore-reproduce checks
5. `src/train.py`, `scripts/run_all.py` — teacher first, gate check, then the rest
6. `src/evaluate.py`, `scripts/aggregate.py` — only after all training completes

---

## 4. Component specifications

### 4.1 Data (`src/data.py`)

```python
SPLIT_SEED = 20260828  # fixed for the entire study; NOT the run seed
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
```

- Load CIFAR-10 train (50k) and test (10k) as distributed.
- Partition train into 45k/5k using a `numpy.random.RandomState(SPLIT_SEED)` permutation. The split must be byte-identical across every run and every arm. Assert this by hashing the index arrays and comparing against a constant stored in the module.
- Train transform: `RandomCrop(32, padding=4)` (zero padding) → `RandomHorizontalFlip(p=0.5)` → `ToTensor` → `Normalize`.
- Val and test transform: `ToTensor` → `Normalize` only.
- **The teacher trains on the same 45k.** No exceptions.

### 4.2 Models (`src/models.py`)

CIFAR-adapted ResNet-18/34: standard torchvision-style BasicBlock ResNet, with the 7×7 stride-2 stem replaced by 3×3 stride-1 padding-1, and the initial maxpool removed. Everything else unchanged. Layer configuration [2,2,2,2] for R18, [3,4,6,3] for R34, channels 64/128/256/512.

6n+2 family for arm E: He et al. CIFAR architecture, 3 stages of 2n BasicBlocks at 16/32/64 channels, 3×3 stem, global average pool, 10-way FC. n ∈ {3,5,7,9,18} → ResNet-20/32/44/56/110.

Initialization:
- Conv: `kaiming_normal_(w, mode='fan_out', nonlinearity='relu')`
- BatchNorm: `γ = 1`, `β = 0` for **every** BN including the last in each residual block
- FC: PyTorch default

**Do not zero-initialize the final BN γ in residual blocks.** This is a deliberate omission (pre-registration §12).

### 4.3 Quantizer (`src/quantizer.py`)

TWN closed-form, **per output filter**. For a conv weight of shape `(out_ch, in_ch, kh, kw)`, filter *k* is `W[k]` and `n = in_ch * kh * kw`.

```python
def ternarize(w: Tensor) -> tuple[Tensor, Tensor]:
    """
    w: (out_ch, in_ch, kh, kw) latent weights
    returns: (w_ternary, degenerate_mask)
      w_ternary: same shape, values in {-alpha_k, 0, +alpha_k} per filter k
      degenerate_mask: (out_ch,) bool, True where the fallback fired
    """
    out_ch = w.shape[0]
    w_flat = w.view(out_ch, -1)                    # (out_ch, n)
    n = w_flat.shape[1]

    delta = 0.75 / n * w_flat.abs().sum(dim=1, keepdim=True)   # (out_ch, 1)
    mask = w_flat.abs() > delta                                 # strict >

    # Degenerate fallback: empty support -> force the largest-magnitude weight
    degenerate = ~mask.any(dim=1)                               # (out_ch,)
    if degenerate.any():
        idx = w_flat[degenerate].abs().argmax(dim=1)
        rows = degenerate.nonzero(as_tuple=True)[0]
        mask[rows, idx] = True

    signs = w_flat.sign() * mask                                # {-1, 0, +1}
    alpha = (w_flat.abs() * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)

    return (alpha * signs).view_as(w), degenerate
```

Notes:
- Threshold comparison is **strict** `>`, matching TWN Eq. 3.
- `alpha` is the mean magnitude over the surviving set. With the fallback, that is the single surviving weight's magnitude — still the L2-optimal α for that support.
- Recomputed **on every forward pass**, from current latent weights. Do not cache per-epoch.
- Assert `mask.sum(dim=1) > 0` after the fallback.

### 4.4 Ternary conv module (`src/ternary_modules.py`)

```python
class TernaryConv2d(nn.Conv2d):
    def forward(self, x):
        w_t, degen = ternarize(self.weight)
        self._last_degenerate = degen.detach()          # for logging
        w_ste = self.weight + (w_t - self.weight).detach()   # vanilla identity STE
        return F.conv2d(x, w_ste, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
```

The STE line is the whole mechanism: forward uses `w_t`, backward passes gradient straight to `self.weight` unchanged. Do not multiply by any mask or scale — that would be a different STE (pre-registration §12 forbids it).

**Latent clamping** happens in the training loop after `optimizer.step()`, not here:

```python
for m in model.modules():
    if isinstance(m, TernaryConv2d):
        m.weight.data.clamp_(-1, 1)
```

**Which layers get wrapped:** all conv layers **except** `model.conv1` (the stem). This includes 1×1 downsample convs in shortcut paths. The final FC stays `nn.Linear` in fp32. Build the model normally, then walk it and swap; assert the resulting count of `TernaryConv2d` modules matches an expected constant.

### 4.5 KD loss (`src/losses.py`)

```python
def kd_loss(student_logits, teacher_logits, targets, T=4.0, lam=0.9):
    hard = F.cross_entropy(student_logits, targets)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean',
    )
    return (1 - lam) * hard + lam * (T ** 2) * soft
```

- `T**2` sits **inside** the soft term, before `lam`. Effective soft:hard gradient ratio is `lam/(1-lam)` = 9:1.
- `reduction='batchmean'`, not `'mean'`. `'mean'` additionally divides by the class count and silently rescales the loss by 10×.
- PyTorch's `kl_div(input, target)` computes KL(target ‖ input) with `input` as log-probabilities, which is KL(teacher ‖ student) — correct.
- Teacher runs in `eval()` mode under `torch.no_grad()`, on **the same augmented batch** the student sees. Do not precompute teacher logits; augmentation is stochastic.

### 4.6 Tracking (`src/tracking.py`)

Three per-layer metrics, logged every epoch.

**Zero fraction** — fraction of a ternarized layer's weights quantized to 0. Computed from the current `ternarize()` output at end of epoch. Full-precision layers excluded.

**Degenerate filter count** — sum of `degenerate` mask per layer, captured from the last forward pass of the epoch.

**Reversal rate** — the error-prone one. Read this carefully.

Maintain two persistent buffers per ternarized layer, same shape as the weight:
- `prev_state` ∈ {−1, 0, +1}, the ternary assignment at the end of the previous epoch
- `last_change_dir` ∈ {−1, 0, +1}, the *direction* of that weight's most recent state change, ever; 0 means "has never changed"

At the end of each epoch:

```python
cur_state = signs_of_current_ternarization        # {-1, 0, +1}, same shape as weight
changed   = cur_state != prev_state
direction = torch.sign(cur_state - prev_state)    # {-1, 0, +1}

reversal = changed & (last_change_dir != 0) & (direction != last_change_dir)

reversal_rate = reversal.sum() / cur_state.numel()   # denominator: ALL weights in layer

last_change_dir = torch.where(changed, direction, last_change_dir)
prev_state = cur_state
```

Critical points, each of which changes the number if got wrong:

- **The comparison is against `last_change_dir`, not against the previous epoch's direction.** A weight stable for twenty epochs then changing is compared against its last actual change, however long ago.
- **Denominator is all weights in the layer**, not only those that have ever changed. This keeps the metric comparable across epochs and arms.
- **Direction is the sign of the numeric state difference.** So −1 → 0 → +1 is monotone (not a reversal); 0 → +1 → 0 is a reversal; −1 → +1 → 0 is a reversal.
- **Reversal rate is `null` for epochs 1 and 2.** At epoch 1 there is no previous state; at epoch 2 `last_change_dir` is still all zero so no reversal is detectable. Emit JSON `null`, not `0.0` — a zero would be indistinguishable from a genuinely stable epoch.
- Log **per layer**, plus an aggregate over all ternarized weights (weighted by weight count, i.e. total reversals / total ternarized weights).

### 4.7 Storage accounting (`src/storage.py`)

Fixed-width 2-bit encoding. No entropy coding, no base-3 packing, no sparse formats, **no BN folding**.

Ternary model exported storage, in bits:
```
  sum over ternarized convs: n_weights * 2  +  n_out_filters * 32     # weights + per-filter alpha
+ first conv params * 32
+ final FC params (weight + bias) * 32
+ all BN gamma, beta, running_mean, running_var * 32                   # shipped, not folded
```

fp32 model exported storage, in bits:
```
  all parameters * 32
+ all BN buffers (running_mean, running_var) * 32
```
(BN γ and β are already in `parameters()`; running stats are buffers and must be added separately.)

**Arm E selection** (`scripts/select_arm_e.py`) — run once, before any training, output written into the run manifest and never revisited:

```
target = ternary_resnet18_storage_bits()
candidates = {20: r20, 32: r32, 44: r44, 56: r56, 110: r110}
best = argmin over candidates of |storage(c) - target|
# tie-break: if two candidates' |storage - target| / target differ by < 5%,
#            take the one with fewer parameters
```

This is computable from architecture alone — that is why the fixed-width convention is mandatory. If you find yourself needing a trained model's zero fraction to compute storage, you have deviated.

### 4.8 Training loop (`src/train.py`)

```python
EPOCHS = 160
LR = 0.1
MILESTONES = [80, 120]     # LR drops BEFORE these epochs begin (1-indexed)
GAMMA = 0.1
BATCH_SIZE = 128
MOMENTUM = 0.9             # plain, NOT Nesterov
WEIGHT_DECAY = 1e-4
CHECKPOINT_EPOCHS = range(151, 161)
```

Epoch indexing is **1-based**. LR is 0.1 for epochs 1–79, 0.01 for 80–119, 0.001 for 120–160. If you use `MultiStepLR`, verify the boundary matches this exactly; off-by-one here shifts every number.

**Weight decay param groups:**
- Decay (1e-4): all conv weights (latent weights for ternarized layers, direct for the fp stem) and the FC weight
- No decay: all BatchNorm γ and β, and all biases

**Per-step order** — this order is load-bearing:
1. forward (quantization happens inside `TernaryConv2d.forward`)
2. loss, backward
3. `optimizer.step()`
4. clamp latent weights to [−1, 1]

**Determinism setup**, at the start of every run:
```python
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
DataLoader: `num_workers=4`, `shuffle=True`, seeded `generator=torch.Generator().manual_seed(seed)`, and a `worker_init_fn` seeding each worker from `seed + worker_id`.

**fp32 throughout.** No `torch.cuda.amp`, no `bf16`, no `torch.compile` with anything that changes numerics.

**Per epoch:** train, evaluate on val, compute tracking metrics, append one line to `metrics.jsonl`. Save a checkpoint only for epochs 151–160.

**Never evaluate on test inside the training loop.**

### 4.9 Evaluation (`src/evaluate.py`)

Runs once per arm, after all that arm's training completes.

- Load each of the 10 checkpoints from epochs 151–160, evaluate on test, record 10 accuracies.
- `last10_mean` = arithmetic mean of those 10 **accuracies**. Not an average of weights. Do not implement weight averaging.
- `epoch160_accuracy` = the 160th checkpoint's test accuracy.
- `last10_val_spread` = max − min of validation accuracy over epochs 151–160, from `metrics.jsonl`, in percentage points.
- `settledness_flag` = `last10_val_spread > 0.5`.

---

## 5. Run orchestration (`scripts/run_all.py`)

Strict order. Do not parallelize across these phases.

**Phase 0 — pre-flight.** Run `scripts/verify.py` (§9) and `tests/`. Abort on any failure.

**Phase 1 — arm E selection.** Run `scripts/select_arm_e.py`, write the chosen architecture and both storage figures into a committed constants file. This must happen before training so it cannot be influenced by results.

**Phase 2 — teacher.** Train arm T, seed 0.

> **Teacher gate.** If T's last-10-epoch mean **validation** accuracy < 92%, halt. Treat as a pipeline bug and investigate. **Do not reconfigure, retune, or reseed the teacher to pass.** If a correctly-trained teacher genuinely lands below 92%, record that and proceed with the weak teacher noted as a limitation.

The 92% figure is a smoke test for broken training, not a quality bar (He et al. report 91.25% for a 0.27M-param ResNet-20). It is **not** a comparison against any student arm; that comparison happens at the end, on test, in the analysis.

**Phase 3 — headline arms.** A (seeds 0,1,2), B (seeds 0,1), C (seeds 0,1,2), D (seeds 0,1,2), E (seed 0). These may run in any order or in parallel.

**Phase 4 — evaluation.** `src/evaluate.py` for every arm. This is the first and only test-set access.

**Phase 5 — sweep.** Only after Phase 4 numbers are written and committed. Arm D at T=2 (seeds 0,1) and T=8 (seeds 0,1). The sweep cannot revise any Phase 3/4 number.

**Seeds are per arm**, drawn in order from [0, 1, …, 15]. Seed 0 of arm A and seed 0 of arm C are unrelated runs.

---

## 6. Discard rule

A run is discarded **only** for:

1. NaN or Inf in the training loss at any step
2. Mean training loss over epoch 160 > mean training loss over epoch 1
3. Infrastructure failure (crash, OOM, preemption, corrupted checkpoint)

**A run that trains cleanly to a poor accuracy is a result, not a failure.** Never discard on accuracy. Nothing in the discard path may read val or test accuracy.

On discard: mark `discard_status` in the manifest, **retain all logs**, and draw the next seed in sequence. Report discard counts **per arm**. If any arm exceeds **3 discards**, halt and report the configuration as unstable for that arm rather than continuing to draw seeds.

---

## 7. Analysis (`scripts/aggregate.py`)

Every reported number is produced by this script from files in `runs/`. Nothing is computed by hand.

**Spread means max − min**, in percentage points, everywhere — across seeds and across epochs. Arm B's spread comes from 2 seeds; every table cell and sentence using B's spread must carry that annotation.

Fixed interpretation rules, implement them as code and let them emit the verdict:

```python
max_spread = max(spread(A), spread(B), spread(C), spread(D))   # includes B

# Recovery ratio
if C_mean >= A_mean or abs(A_mean - C_mean) < max_spread:
    R = None
    verdict = "undefined; report raw accuracies, no 'recovery' language"
else:
    R = (D_mean - C_mean) / (A_mean - C_mean)

# Interaction
interaction = (D_mean - C_mean) - (B_mean - A_mean)
if abs(interaction) > 2 * max_spread:
    verdict = "suggestive (single-seed-limited, not an effect size)"
else:
    verdict = "unresolved"

# Teacher comparison
if T_test <= A_mean:
    verdict = "B and D uninterpretable as distillation; interaction not computed"
```

Any difference smaller than the largest relevant within-arm spread is reported as **not resolved**, never as a small effect. Single-seed arms (E, and the sweep points) yield threshold judgments and directional statements only.

**Output table columns:** arm, seeds, last-10 test mean, spread, epoch-160 test, settledness flag, discard count, parameters, bits/weight, exported storage MB, MACs.

Also emit: per-layer zero fraction (last-10 mean), per-layer degenerate filter count (last-10 mean and max over training), per-layer and aggregate reversal rate curves, validation learning curves for all arms.

**Compression framing in the writeup:** lead with matched storage; state in the same paragraph that MACs are unchanged by weight ternarization and that no latency claim follows without custom kernels. Checkpoints hold fp32 latent weights and are full-size; compression describes the exported model only.

---

## 8. Logging schema

Per run, under `runs/{arm}_{seed}/`:

**`manifest.json`** (written at run start)
```json
{
  "arm": "D", "seed": 0, "split_seed": 20260828,
  "git_commit": "...", "hostname": "...", "gpu_model": "...",
  "torch_version": "...", "cuda_version": "...",
  "start_time": "...", "end_time": "...",
  "hyperparameters": { "epochs": 160, "lr": 0.1, "milestones": [80,120],
                       "batch_size": 128, "momentum": 0.9, "nesterov": false,
                       "weight_decay": 1e-4, "T": 4.0, "lambda": 0.9,
                       "clamp": [-1,1], "ste": "identity" },
  "architecture": "resnet18_cifar",
  "ternarized_layer_names": ["layer1.0.conv1", "..."],
  "discard_status": "completed"
}
```

**`metrics.jsonl`** (one object per epoch)
```json
{"epoch": 3, "lr": 0.1, "train_loss_mean": 1.42, "train_loss_final_step": 1.38,
 "val_accuracy": 71.3, "epoch_wall_time": 41.2,
 "per_layer": {"layer1.0.conv1": {"zero_fraction": 0.41, "reversal_rate": 0.021,
                                  "degenerate_filter_count": 0, "n_weights": 36864}},
 "aggregate_reversal_rate": 0.019}
```
`reversal_rate` and `aggregate_reversal_rate` are `null` for epochs 1 and 2. Ternary-only fields are omitted entirely for fp arms.

**`checkpoints/epoch_{151..160}.pt`** — full `state_dict` including BatchNorm buffers. Epochs 1–150 are not checkpointed. Ternary arms save fp32 **latent** weights; requantization at load is deterministic.

**`test_results.json`** (written once, after the arm completes)
```json
{"per_checkpoint_test_accuracy": [93.1, "...10 values..."],
 "last10_mean": 93.14, "epoch160_accuracy": 93.2,
 "last10_val_spread": 0.31, "settledness_flag": false}
```

---

## 9. Pre-flight verification (`scripts/verify.py` + `tests/`)

All must pass before Phase 2.

**Quantizer**
- Output of `ternarize` contains at most 3 distinct values per filter, and they are {−α_k, 0, +α_k}
- α equals the mean magnitude of weights with |w| > Δ, to float tolerance
- Hand-constructed all-below-threshold filter → exactly one surviving weight, α = its magnitude, `degenerate` True
- Δ uses filter-level `n`, not layer-level: construct a conv with known weights and assert Δ against a hand computation

**STE**
- `w_ste.backward()` gives `weight.grad` exactly equal to the upstream gradient, elementwise (identity passthrough, no masking)

**KD loss**
- `lam=0` reproduces plain cross-entropy exactly
- Identical student and teacher logits → soft term is 0
- Soft-term gradient magnitude w.r.t. student logits is approximately equal at T=1 and T=4 (this is what the T² factor buys; if it differs by ~16× the factor is misplaced)
- `reduction='batchmean'` — assert the loss does not change when the number of classes changes but the distributions do not

**Reversal rate** — drive `tracking` with a hand-built state sequence and assert against expected counts:
```
weight 1: +1 +1 +1 +1   -> 0 reversals
weight 2:  0 +1  0 +1   -> 2 reversals (at epochs 3 and 4)
weight 3: -1  0 +1 +1   -> 0 reversals (monotone)
weight 4: -1 +1  0  0   -> 1 reversal (at epoch 3)
weight 5:  0 +1 +1  0   -> 1 reversal (at epoch 4; compares to the change at epoch 2)
```
Epochs 1 and 2 must return `None`.

**Storage** — recompute the ternary R18 and each 6n+2 candidate's storage by hand once, assert against `storage.py`. Assert BN buffers are included and no folding is applied.

**Determinism** — run two 2-epoch trainings with the same seed on the same machine; assert identical loss at step 1, identical loss at the end of epoch 1, and identical val accuracy at epoch 2. Then run with a different seed and assert they differ.

**Restore-and-reproduce** — on the teacher run: reload a saved checkpoint, evaluate on **validation**, assert it exactly reproduces that checkpoint's logged validation accuracy. Test is not involved.

**Data hygiene** — assert the test loader is never instantiated by `src/train.py`. A grep-level check in CI is fine.

---

## 10. Do not do these

Each of these was considered and rejected. Adding any of them invalidates the study.

| Tempting | Why not |
|---|---|
| Cosine LR schedule | Step schedule is what the cited papers use; cosine was rejected as an unforced deviation |
| Train longer than 160 epochs | N is fixed and compute-bounded; changing it after the fact is selection |
| Save "best" checkpoint by val accuracy | Checkpoint selection was deliberately removed; it biases toward noisier arms |
| Report best-epoch accuracy | Same reason; primary metric is the last-10 mean |
| Tune T, λ, LR, or weight decay | Zero hyperparameters may be selected from data, for any arm |
| Per-arm hyperparameters | Breaks search parity |
| Zero-init final BN γ | Postdates the cited configs; effect may differ between fp and ternary arms |
| AMP / bf16 / TF32 | fp32 only; mixed precision interacts with the quantizer |
| Nesterov momentum | Plain momentum specified |
| Gradient clipping | None specified |
| Learned or asymmetric α (TTQ-style) | Closed-form TWN per-filter α only |
| Per-layer instead of per-filter α | Per-filter, with filter-level `n` |
| Clipped/saturating STE | Vanilla identity STE + clamping; the combination was chosen deliberately |
| Fold BN at export | BN ships separately; both params and buffers count toward storage |
| Entropy coding / base-3 packing / sparse storage | Would make storage depend on trained weights and retroactively change arm E |
| BN recalibration on headline numbers | Available as a diagnostic only, if a run fails settledness |
| Attention transfer / FitNet hints / any feature KD | Would confound the interaction term |
| Warm-start the ternary arms from a trained fp model | All arms train from scratch |
| Precompute teacher logits | Augmentation is stochastic; logits must be computed online |
| Weight averaging over the last 10 checkpoints | The last-10 mean averages *accuracies*, not weights |
| Discard a run because its accuracy looks wrong | Only the three mechanical criteria in §6 |
| Run the sweep before the headline arms | Ordering is binding |
| Compare absolute numbers to published CIFAR figures | Different architecture, data split, and metric |

---

## 11. If you find a genuine problem

Some things in the pre-registration may be wrong in ways nobody caught. If you hit one:

1. **Stop. Do not fix it silently.**
2. Report it with: what you found, which section it affects, what you believe the correct behaviour is, and whether it changes numbers or only readability.
3. Wait for a decision.
4. If the decision is to change something, it goes in the **amendment log** at the end of `PREREGISTRATION.md`, dated, with the reason — append-only, never by editing the frozen text.

Classes of thing worth escalating: an internal contradiction between two sections; a specification that is genuinely ambiguous after reading §4 of this document; a numerical instability that makes a run fail for reasons the discard rule does not cover; a mismatch between this handoff and the pre-registration.

Not worth escalating: something that looks suboptimal. Most of those are deliberate and are recorded in the pre-registration's limitations section.

---

## 12. Compute estimate

17 runs × 160 epochs. On a single modern GPU, CIFAR-10 ResNet-18 at batch 128 runs roughly 15–25 s/epoch for fp arms; ternary arms are somewhat slower because quantization is recomputed every forward pass, and KD arms add a teacher forward pass. Budget roughly 1–2 GPU-hours per run, so **20–35 GPU-hours total**, plus evaluation.

The ternarize call is the main avoidable cost. It is a few reductions over the weight tensor per conv per step — keep it on-GPU, avoid `.item()` or host syncs inside the forward, and do not add logging inside the forward path. Tracking metrics are computed once per epoch, not per step.
