# Pre-Registration: Ternary Weight Quantization with Knowledge Distillation on CIFAR-10

**Frozen:** 2026-09-14

**Status:** No section may be edited after the first training run begins. Deviations are recorded in the append-only amendment log at the end, with date and reason.

---

## 0. Conventions

These apply throughout and resolve terms used in later sections.

- **Spread** means **maximum minus minimum**, in percentage points. Used for both across-seed spread and within-run across-epoch spread. Applied uniformly, including to arm B, whose spread is computed from 2 seeds rather than 3; **every table and statement using B's spread notes the seed-count asymmetry.**
- **Epochs are 1-indexed.** Training runs epochs 1 through 160. "Epochs 151–160" means the final ten.
- **LR milestones fire before the named epoch begins.** The learning rate is 0.1 for epochs 1–79, 0.01 for epochs 80–119, 0.001 for epochs 120–160.
- **Accuracies are averaged, never weights.** Any "mean over the final 10 epochs" is the arithmetic mean of ten separately computed accuracies.
- **Split seed = 20260828**, used once to partition the 45k/5k train/validation split. Fixed for the entire study, identical for every run.

---

## 1. Claim under test

We measure the accuracy effect of constraining ResNet-18's convolutional weights (excluding the first convolution) to {−α, 0, +α} on CIFAR-10, and the extent to which knowledge distillation from a full-precision ResNet-34 teacher modifies that effect, **in either direction**.

We do not presuppose that ternarization costs accuracy. Prior work reports ternary CIFAR ResNets matching or exceeding full precision at comparable or smaller scale, and ResNet-18 is heavily overparameterized for this dataset.

---

## 2. Data

- CIFAR-10, 50k train / 10k test as distributed.
- Train split partitioned 45k/5k using split seed 20260828 (§0). The 5k is the validation split.
- **Every model, including the teacher, trains on the same 45k.** No model sees the 5k during training.
- Validation is evaluated **every epoch, in the training loop**. It is used for learning curves, the settledness criterion, the teacher gate, and the restore-and-reproduce check. It is **not** used for checkpoint selection or hyperparameter tuning.
- **The test set is evaluated only after all training for a given arm is complete**, from saved checkpoints. No component of the design reads test accuracy during training.
- Augmentation: 4-pixel zero pad, random 32×32 crop, random horizontal flip (p=0.5), then normalization with mean (0.4914, 0.4822, 0.4465) and std (0.2470, 0.2435, 0.2616). Validation and test receive normalization only. Identical across all arms.

---

## 3. Architecture

- **Student:** ResNet-18, CIFAR-adapted — 3×3 stride-1 stem replacing the 7×7 stride-2 conv, initial maxpool removed, all else per He et al. (2016). The adaptation is standard practice and is not from that paper.
- **Teacher:** ResNet-34, same CIFAR adaptation.
- **Ternarized layers:** all convolutional layers **except the first**, including the 1×1 downsample convolutions in shortcut paths.
- **Full precision:** first convolution, final fully-connected layer, all BatchNorm parameters and buffers. TTQ excludes first/last on ImageNet but ternarizes the classifier on CIFAR; TWN states no exception. This is our choice, not inherited.
- **Initialization:** `kaiming_normal_`, `mode='fan_out'`, `nonlinearity='relu'` for all conv layers; BatchNorm γ=1, β=0 for **all** blocks including the last in each residual block. Zero-γ initialization is **not** used (see §12).

---

## 4. Quantizer

TWN closed-form, per output filter, following TWN Algorithm 1.

For output filter *k* of layer *m*, with *n* = **the number of weights in that filter** (not the layer):

- Δ_mk = (0.75 / n) · ‖W_mk‖₁
- W̃_i = +1 if W_i > Δ; −1 if W_i < −Δ; 0 otherwise
- α_mk = mean of |W_i| over the surviving set I = {i : |W_i| > Δ}

**Recomputed on every forward pass**, from the current latent weights.

**Degenerate filters.** If I is empty, the weight of largest magnitude in the filter is forced into the surviving set with its own sign, and α is that weight's magnitude. This preserves the L2-optimal α for the resulting support while preventing the filter from dying: an all-zero filter contributes nothing to the forward pass, receives no useful gradient, and its latent weights only shrink further under weight decay, making it an absorbing state. **A per-layer count of degenerate filters is logged every epoch and reported.**

**Per-step order of operations:**

1. Compute Δ, α, W̃ from latent weights
2. Forward pass using α·W̃
3. Backward pass; vanilla (identity) STE through the quantizer
4. Optimizer step on latent weights (including weight decay and momentum)
5. Clamp latent weights to [−1, 1]

Not used: learned scales, asymmetric α_p/α_n, per-layer α, fixed-sparsity thresholds. TTQ report that a fixed threshold outperforms a fixed sparsity target; we do not re-test this.

**Note on the source:** TWN Eq. 6 prints `arg min` where substituting the optimal α yields a quantity to be maximized. We implement the maximization. The reference is arXiv v3 (Nov 2022), whose Algorithm 1 is per-filter with constant 0.75; TTQ cites the earlier version as layer-wise with 0.7.

---

## 5. Training

Identical for every arm unless stated.

| | |
|---|---|
| Initialization | Per §3. **No warm start.** No arm receives pretraining. |
| Epochs | 160 |
| LR schedule | 0.1; ×0.1 before epochs 80 and 120 (see §0) |
| Optimizer | SGD, momentum 0.9, **not** Nesterov |
| Batch size | 128 |
| Weight decay | 1e-4, applied to all conv and FC weights (latent weights for ternarized layers; direct weights for the fp first conv and final FC). **Excluded** from BatchNorm parameters and from all biases. |
| Gradient clipping | None |
| Precision | fp32 throughout. No automatic mixed precision. |
| Latent clamping | [−1, 1] after each optimizer step |

**Knowledge distillation loss** (arms B and D only):

**L = (1 − λ) · CE(z_s, y) + λ · T² · KL( softmax(z_t / T) ‖ softmax(z_s / T) )**

with **λ = 0.9**, **T = 4**, KL computed with `reduction='batchmean'`. The T² factor sits **inside** the soft term, before λ, following Hinton et al.: it compensates the 1/T² shrinkage of the softened-softmax gradient so that λ means the same thing at any T. The resulting effective soft-to-hard gradient ratio is λ/(1−λ) = **9:1**.

**Parameterisation note.** This is the convex form (1−λ, λ). Hinton et al.'s speech experiments used "a relative weight of 0.5 on the cross-entropy for the hard targets" against an unscaled soft term, a 2:1 ratio in a different parameterisation. λ=0.9 is the later-standard convention and is **not** Hinton's own value; it is not tuned.

**Teacher logits are computed online**, per batch, on the same augmented inputs the student sees, with the teacher in `eval()` mode (BatchNorm using running statistics) and under `torch.no_grad()`. Logits are not precomputed, since augmentation is stochastic.

**Determinism.** Each run sets `torch.manual_seed`, `numpy.random.seed`, and `random.seed` to the arm seed; sets `torch.backends.cudnn.deterministic = True` and `benchmark = False`; and uses a DataLoader with `num_workers=4`, `shuffle=True`, and a seeded `generator` plus `worker_init_fn` so that shuffling and augmentation are reproducible. The arm seed controls initialization, data ordering, and augmentation sampling. It does **not** control the 45k/5k split, which is fixed at 20260828 for all runs.

**Justification for fixed values, stated as conventions rather than optima:**

- Weight decay 1e-4 sits in the range used across reference implementations (He et al. 1e-4 on CIFAR and ImageNet ResNets; TWN 1e-4; TTQ 2e-4). Not tuned; not varied by arm.
- Schedule shape and N sit inside the range used by TTQ (milestones 80/120, convergence claimed ~160) and Yin et al. (200 epochs, milestones 80/140) on CIFAR ResNets. **We do not attribute this schedule to either paper:** TTQ's printed milestones include epoch 300 alongside a claim of convergence at 160, which is internally inconsistent. N=160 is compute-bounded.
- T=4, λ=0.9 are conventional. **Not claimed optimal for a ternary student.** Hinton's finding that smaller students prefer lower temperatures concerns MNIST MLPs varying width, not precision, and does not transfer. §9 addresses this separately.

**Zero hyperparameters are selected from data, for any arm, at any point.** No pilots. No sweeps feeding the main comparison.

---

## 6. Arms

**Teacher gate.** One teacher is trained under the fixed §5 configuration. If its **last-10-epoch mean validation accuracy is below 92%**, this is treated as a pipeline failure, investigated as a bug, and the study does not proceed until a teacher trains successfully under the unchanged configuration. The teacher is never reconfigured, retuned, or reseeded to pass. If a correctly-trained teacher lands below 92%, that is reported and the study proceeds with the weak teacher noted as a limitation.

The 92% figure is an absolute smoke test, not a quality bar: He et al. report 91.25% test accuracy for a 0.27M-parameter ResNet-20 on the full 50k split, so a ResNet-34 below 92% indicates a broken pipeline rather than a marginal teacher. **It is not a comparison against any student arm.** Whether the teacher outperformed arm A is a §8 reported result, computed at the end.

Headline arms:

| Arm | Model | Precision | Supervision | Seeds |
|---|---|---|---|---|
| T | ResNet-34 | fp32 | CE | 1 |
| A | ResNet-18 | fp32 | CE | 3 |
| B | ResNet-18 | fp32 | CE + KD from T | 2 |
| C | ResNet-18 | ternary | CE | 3 |
| D | ResNet-18 | ternary | CE + KD from T | 3 |

Directional check (1 seed, marked as such in every table):

| Arm | Description | Seeds |
|---|---|---|
| E | fp32 net at matched storage to ternary R18 | 1 |

**E's architecture, computed once before any training.** Candidates: the He CIFAR 6n+2 family with 16/32/64 filters — ResNet-20, 32, 44, 56, 110. Select the member minimizing |storage_candidate − storage_target| where storage_target is the ternary ResNet-18's exported storage under the §7 accounting convention and storage_candidate is that member's parameter count × 32 bits plus its BatchNorm parameters and buffers. Ties resolve to the smaller network when |storage_candidate − storage_target| / storage_target differs by less than 5% between two members. The selected member and both storage figures are recorded in the run manifest before training begins and are not revisited.

**Not run, with reasons:** label-smoothing control (would land inside noise at n=1; addressed by discussion). Full-budget warm-start control (obviated by scratch training). Feature-level KD arms (would confound the interaction term). Matched-FLOPs control (arm A serves this role, since weight ternarization leaves MAC count unchanged).

**What each arm rules out:** T — that a weak result reflects a weak teacher. A — nothing; it is the full-precision denominator. C — that KD did the work. B — that any KD benefit is generic rather than specific to ternary students. E — that a smaller float model would have achieved the same.

**Seeds are drawn per arm** from the sequence [0, 1, 2, …, 15], starting at 0 and drawing upward as needed. Arms use the same seed values independently: seed 0 for arm A and seed 0 for arm C are different runs. Sixteen seeds per arm accommodates the §10 discard cap with margin.

---

## 7. Measurement and reporting

**Primary metric:** mean test accuracy over the **ten saved checkpoints from epochs 151–160**, each loaded and evaluated once on the test set after training completes, with the ten resulting accuracies averaged. Secondary: epoch-160 test accuracy, reported in the same table.

Rationale: max-over-epochs selection is biased upward in proportion to an arm's variance, and the ternary arms are expected to be noisier, so best-epoch reporting would inflate them in the direction of the claim. TTQ report a moving average over all epochs to filter fluctuation without specifying a window; we adopt the principle with an explicit fixed window.

**Reported for every arm:**

- Mean and spread (§0) across seeds. Never mean alone. For arm B, the seed-count asymmetry (2 vs 3) is noted wherever its spread appears.
- **Zero fraction:** per ternarized layer, computed as the fraction of that layer's weights quantized to 0, **averaged over epochs 151–160**. Full-precision layers are excluded.
- **Degenerate filter count:** per ternarized layer, logged every epoch, reported as the last-10 mean and the maximum over training.
- Parameters, bits/weight, exported storage in MB, and MACs.
- Validation learning curves (per-epoch validation accuracy, all seeds).
- Assignment reversal rate, per specification below.

**Reversal rate.** Logged **every epoch**, computed **per ternarized layer** and additionally aggregated over all ternarized weights.

- A weight's *state* is its ternary assignment in {−1, 0, +1}, recorded at the end of each epoch.
- A *change* is any epoch where the state differs from the previous epoch. Its *direction* is the sign of (new state − old state).
- A *reversal* is a change whose direction is opposite in sign to that weight's **most recent prior change, regardless of how many epochs earlier it occurred.** A weight stable for twenty epochs then changing still compares against its last change, not against epoch t−1.
- **Denominator: all ternarized weights in the layer**, not only those that have ever changed, so the metric is comparable across epochs and arms.
- Reversal rate is defined from **epoch 3 onward**, since detecting one requires at least two prior changes to compare.

Epoch granularity is chosen because per-step logging would capture minibatch jitter that is not meaningful, coarser logging would alias faster oscillations into invisibility, and epoch resolution matches the validation-accuracy series so the two are directly comparable.

**Storage accounting convention.** All storage figures assume **fixed-width 2-bit encoding** of ternary weights, plus one fp32 scale factor per output filter per ternarized layer (per §4), plus fp32 parameters for the first convolution and final FC, plus **BatchNorm γ, β, running mean, and running variance in fp32**. BatchNorm is **shipped separately, not folded into preceding convolutions**, so all four BN tensors count toward exported storage and the verification script checks convolutional weights as saved rather than as folded. No entropy coding, base-3 packing, or sparse formats are used or claimed. This convention is fixed in advance so that exported storage is a function of architecture alone and is computable before training, which is what makes E's selection pre-computable.

Zero fraction is reported as a measured property. Entropy coding or base-3 packing would reduce storage below the fixed-width figure; **we do not compute or claim a compressed size**, because doing so would make storage depend on trained weights and would retroactively affect E's selection.

**Compression reporting:** matched-storage leads, since that is the claim weight quantization makes. Stated in the same paragraph: MACs are unchanged by weight ternarization, and no latency claim follows without custom kernels, which we did not write. Checkpoints store fp32 latent weights and are full-size; compression describes the exported model only.

**Settledness criterion.** A run is flagged as unsettled if the spread (§0) of its **validation** accuracy over epochs 151–160 exceeds **0.5 percentage points**. The threshold is a declared round number, not calibrated from data.

This is a **weak criterion**: it catches runs whose final-phase variance makes a single reported number unreliable, and nothing finer. An earlier two-part version comparing the final window against the pre-LR-drop window was considered and rejected, because a run noisy at the end is generally noisy earlier, making the second condition near-vacuous. Reversal rate is reported alongside as a diagnostic that may explain a flagged run, but is not itself a pass/fail test.

**Restore-and-reproduce check.** Run once, before the main study, on the teacher run. A saved checkpoint is reloaded and evaluated **on the validation split**; it must reproduce that checkpoint's logged validation accuracy exactly. Test is not involved. Checkpoints store the full `state_dict` including BatchNorm buffers.

---

## 8. Interpretation rules, fixed in advance

**Teacher comparison.** T's test accuracy is reported in every table alongside the distillation arms.

> If T's test accuracy is at or below arm A's mean, the teacher provides no capability the student lacks, and arms B and D are reported as uninterpretable as distillation results. The interaction term is not computed. This is a reported outcome discovered at the end, not a gate; the §6 gate is a separate pipeline smoke test.

**Recovery ratio** R = (D − C) / (A − C), on last-10 test means.

> **R is undefined, and no "recovery" language is used, if C ≥ A or if |A − C| is smaller than the largest within-arm seed spread.** In that case raw accuracies are reported and the result is described as ternarization matching or exceeding full precision.

**Interaction** (D − C) − (B − A), the differential KD benefit to the ternary student:

> Reported as *suggestive* only if it exceeds **twice the largest within-arm seed spread among A, B, C, and D** — where that maximum includes B, whose spread comes from 2 seeds and is noted as such. Reported as *unresolved* otherwise. Never reported as a quantified effect size, given B's seed count.

**General:** any difference smaller than the largest relevant within-arm spread is reported as **not resolved**, not as a small effect. Single-seed arms yield threshold judgments and directional statements only, never effect sizes, and are marked as single-seed in every table where they appear.

---

## 9. Temperature sweep

Arm D at T ∈ {2, 4, 8}, 2 seeds per point (seeds 0 and 1), all else identical to §5. Arm C serves as the no-teacher reference line. The T=4 points are the existing D arm runs at seeds 0 and 1; only T=2 and T=8 require new runs.

**Ordering is binding:** the §6 arms are run, completed, and their numbers recorded before any sweep run begins. The sweep cannot revise any §6 number, hyperparameter, or interpretation. If the sweep shows a temperature outperforming T=4 for D, this is reported as an unexploited finding.

**Claim scope:** this characterizes the ternary student's response to temperature. It is **not** a test of the capacity-gap hypothesis, because no matched full-precision KD arm was swept. That question requires B and D over the same grid at ≥2 seeds, which we did not run.

**Reporting:** monotone trends, or differences exceeding the across-seed spread. Everything else is reported as flat.

---

## 10. Discard rule

A run is discarded **only** if it meets one of these mechanical criteria, none of which reads accuracy:

1. NaN or Inf appears in the training loss at any step.
2. **Mean training loss over epoch 160 exceeds mean training loss over epoch 1.**
3. Infrastructure failure: crash, OOM, preemption, corrupted checkpoint.

A run that trains cleanly to a poor number **is a result, not a failure**, and is never discarded.

- Replacements are drawn strictly in order from the per-arm seed sequence. No seed is chosen.
- Discard counts are reported **per arm**, not aggregated.
- Logs and manifests from discarded runs are retained.
- **Cap:** if any arm discards more than 3 runs, the configuration is reported as unstable for that arm and the instability is reported as a finding. Seeds are not drawn until three runs happen to complete.

---

## 11. Logging schema

Every run produces three artifacts under `runs/{arm}_{seed}/`.

**`manifest.json`** — written at run start, one object:

```
arm, seed, split_seed, git_commit, hostname, gpu_model,
torch_version, cuda_version, start_time, end_time,
all §5 hyperparameters, architecture name, ternarized_layer_names,
discard_status ("completed" | "discarded:{criterion}")
```

**`metrics.jsonl`** — one JSON object per epoch:

```
epoch, lr, train_loss_mean, train_loss_final_step,
val_accuracy, epoch_wall_time,
per_layer: { layer_name: { zero_fraction, reversal_rate,
                           degenerate_filter_count, n_weights } },
aggregate_reversal_rate
```

Fields `reversal_rate` and `aggregate_reversal_rate` are `null` for epochs 1 and 2. Ternary-specific fields are omitted for fp arms.

**`checkpoints/epoch_{NNN}.pt`** — full `state_dict` including BatchNorm buffers, for **epochs 151–160 only**. Epochs 1–150 are not checkpointed. Latent (fp32) weights are saved for ternary arms; requantisation at load is deterministic given §4.

**`test_results.json`** — written once, after all training for the arm completes:

```
per_checkpoint_test_accuracy: [10 values],
last10_mean, epoch160_accuracy,
last10_val_spread, settledness_flag
```

Aggregate tables are produced by a single script from these files. No number reported in the writeup is computed by hand.

---

## 12. Explicitly not done

- No checkpoint selection on validation.
- No hyperparameter tuned on any split, for any arm.
- No best-epoch reporting.
- No test-set access before final numbers.
- No warm starting from a pretrained student.
- No feature-level distillation, attention transfer, or intermediate-layer hints.
- No learned, asymmetric, or per-layer scaling factors.
- No per-arm hyperparameter tuning of any kind.
- No BatchNorm recalibration applied to headline numbers. It remains available as a diagnostic if an arm fails the settledness criterion, reported as such.
- **No zero-γ initialization of the final BatchNorm in each residual block.** This trick postdates He et al. (2016) and appears in none of TWN, TTQ, or Yin et al. Its effect could plausibly differ between fp and ternary arms, which would shift the comparison in an unbounded direction, so it is omitted to stay closer to the cited configurations.
- No entropy coding, base-3 packing, or sparse-format storage claims.
- No BatchNorm folding at export.
- No comparison of our absolute numbers to published CIFAR figures. Architecture (ResNet-18 vs. the 6n+2 family), training data (45k vs. 50k), and reporting metric all differ. Reference papers are used for direction and mechanism only.
- No claim about latency or inference speed.

---

## 13. Known limitations, acknowledged in advance

- N=160 is compute-bounded. If ternary arms are undertrained relative to fp arms, the bias runs against the KD claim. Flat validation curves over the final third are offered as evidence against undertraining; a 2× budget control was not run.
- T=4 may not be optimal for a ternary student, which biases against arm D.
- λ=0.9 is a later convention, not Hinton's own value, and is untuned.
- B at 2 seeds limits the interaction term to a directional statement, and its spread is a 2-point range.
- E at 1 seed supports a threshold judgment only.
- Weight decay is not tuned per arm and may not suit ternary training.
- The 45k/5k split costs 10% of training data relative to published setups and may not affect all arms equally; arms with greater effective capacity are expected to lose more.
- The degenerate-filter fallback overrides TWN's support selection for affected filters, though it preserves the L2-optimal α for the resulting support.
- Whether reduced precision constrains the hypothesis space in the same way reduced width does is an assumption, not a result.
- The settledness threshold (0.5 pp) and the teacher gate threshold (92%) are declared round numbers, not empirically calibrated.
- Three seeds per headline arm is the minimum that yields a spread at all; it does not support formal statistical testing, and none is performed.

---

## References

- He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS Workshop 2015.
- Li, Liu, Wang, Zhang, Yan. *Ternary Weight Networks.* arXiv:1605.04711v3 (Nov 2022).
- Zhu, Han, Mao, Dally. *Trained Ternary Quantization.* ICLR 2017.
- Bengio, Léonard, Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* arXiv:1308.3432, 2013.
- Yin, Lyu, Zhang, Osher, Qi, Xin. *Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets.* ICLR 2019.

---

## Amendment log

*(append-only; every entry dated with reason)*

### 2026-09-14 — A1: BatchNorm γ/β double-count in the §6 arm E storage formula

§6 defines a candidate's storage as "parameter count × 32 bits plus its BatchNorm
parameters and buffers". BatchNorm γ and β are already inside `parameters()`, so this
counts them twice. The §7 accounting convention, which §6 is meant to instantiate, counts
each tensor once.

**Resolution:** storage is computed as `all parameters × 32 + BN running_mean and
running_var × 32`. `num_batches_tracked` is excluded, being an int64 step counter rather
than a shipped tensor. **Affects readability and the printed storage figures only
(~0.5% per candidate); it does not change arm E's selection**, which is ResNet-44 under
either reading.

### 2026-09-14 — A2: the 6n+2 family has n BasicBlocks per stage, not 2n

The implementation handoff §4.2 specifies "3 stages of 2n BasicBlocks". A BasicBlock is
two conv layers, so 2n blocks per stage yields 12n+2 layers — ResNet-38 at n=3 — with
roughly double the intended parameter count. The handoff contradicts itself, since its own
mapping n ∈ {3,5,7,9,18} → ResNet-20/32/44/56/110 holds only for n blocks per stage, and
§6 of this document names the 6n+2 family.

**Resolution:** n BasicBlocks per stage. Verified against He et al. (2016), who report
0.27M parameters for ResNet-20; this implementation yields 269,722 at n=3. Under the
handoff's literal text arm E would have been a different and substantially larger network,
so this is a correction, not a preference.

### 2026-09-14 — A3: resume checkpointing, to make §10.3 workable on Colab

This study runs on Google Colab, where session preemption is routine. §10.3 lists
preemption as a discard criterion and §10 caps discards at 3 per arm, while §11
checkpoints only epochs 151–160. A disconnect at, say, epoch 140 would therefore destroy a
run with nothing to resume from, and across 17 runs the arms would predictably hit the
discard cap for reasons unrelated to the configuration under test.

**Resolution:** each run additionally overwrites a single `resume.pt` at the end of every
epoch, holding model, optimizer and LR-schedule state, the Python/NumPy/Torch RNG states,
and the reversal-rate tracking buffers. On restart a run resumes from it. Because the RNG
state is restored, the resumed run is the same run and produces the same numbers; this is
infrastructure, not a change to the experiment. §10.3 continues to apply to crashes, OOM
and corrupted checkpoints, and a preemption that cannot be resumed is still a discard. The
epoch 151–160 archive checkpoints of §11 are unchanged, and `resume.pt` is not used for
any reported number.
