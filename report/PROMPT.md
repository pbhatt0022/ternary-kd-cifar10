# Prompt: write the assignment report

Run this from the repo root **after Phase 4 has completed** and `scripts/aggregate.py` has
produced the final tables. Sections 2, 3 and 4 of the report, plus `make_figures.py`, can be
drafted earlier — they depend only on the code and the frozen documents, not on results.

---

You are writing the final report for a course assignment on ternary weight quantization with
knowledge distillation on CIFAR-10. It is submitted as a PDF and marked out of 30 against the
rubric below.

## Your inputs

Read these before writing anything.

- `PREREGISTRATION.md` — the frozen experimental design, **including its append-only
  amendment log**. Every methodological choice in the report must match it. Where the report
  and this document disagree, the document is right.
- `ATDL_ASSIGNMENT1 Handoff.md` — the implementation spec (note the filename: it contains a
  space, and there is no `HANDOFF.md`). Use it for exact formulas and algorithm descriptions.
- `src/` — read the code, so the method description matches what actually ran. At minimum
  `src/quantizer.py`, `src/losses.py`, `src/tracking.py`, `src/train.py`, `src/storage.py`.
- `report/data/` — the experimental data and aggregate output, fetched from Drive (see below).

### Getting the data (it is not in the repo)

`runs/` lives on Google Drive at `/content/drive/MyDrive/atdl_runs`, not in the repository —
it is gitignored deliberately. Before writing, copy these into `report/data/`:

```
runs/*/manifest.json
runs/*/metrics.jsonl
runs/*/test_results.json
runs/discards.json
runs/teacher_gate.json
runs/report/            <- everything scripts/aggregate.py produced
```

**Do not download `runs/*/checkpoints/`.** That is roughly 45 MB x 10 epochs x 17 runs, about
7.6 GB, and nothing in the report needs it. The files above total a few MB.

### `aggregate.py` runs on Colab, not locally

It imports torch (for parameter counts, exported storage and MACs), and torch cannot load on
the authoring machine — Windows Application Control blocks it. Run **section 9 of the Colab
notebook**, then download `runs/report/`. Its `summaries.json` already holds every
architecture-derived figure the report needs, so nothing is lost by not running it here.

`aggregate.py` produces:

| File | Contents |
|---|---|
| `report.md` | headline table, sweep table, section 8 verdicts, per-layer tables, compression framing, discards |
| `summaries.json` | per-arm means, spreads, parameters, bits/weight, storage MB, MACs, and the verdict objects |
| `val_curves.csv` | `label,epoch,value` — validation accuracy per epoch, every run |
| `reversal_curves.csv` | `label,epoch,value` — aggregate reversal rate per epoch |
| `reversal_curves_per_layer.csv` | `label,layer,epoch,reversal_rate` |
| `zero_fraction_curves.csv` | `label,layer,epoch,zero_fraction` |
| `degenerate_curves.csv` | `label,layer,epoch,degenerate_filter_count` |

## Reference papers

Cite by author and year. Their scope limits matter and are discussed throughout.

- He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS-W 2015.
- Li, Liu, Wang, Zhang, Yan. *Ternary Weight Networks.* arXiv:1605.04711v3 (Nov 2022).
- Zhu, Han, Mao, Dally. *Trained Ternary Quantization.* ICLR 2017.
- Bengio, Leonard, Courville. *Estimating or Propagating Gradients Through Stochastic
  Neurons.* arXiv:1308.3432, 2013.
- Yin, Lyu, Zhang, Osher, Qi, Xin. *Understanding Straight-Through Estimator in Training
  Activation Quantized Neural Nets.* ICLR 2019.

## The rubric (30 marks)

| Component | Marks | Criteria |
|---|---|---|
| Ternary quantizer & student architecture | 6 | Correct ternary quantization formula (alpha, Delta) implemented per-tensor or per-channel; latent FP weights maintained separately from quantized forward weights; STE or other methods correctly implemented and justified; architecture choices (which layers stay FP32) clearly documented. |
| KD formulation & integration | 6 | Correct temperature-scaled soft-target KL loss; sensible combination with hard-label CE; teacher correctly frozen with no gradient leakage; hyperparameters (T, lambda) explored and justified. |
| Joint KD+QAT training pipeline & stability | 7 | Ternary forward pass used consistently during KD training; training is stable (loss curves shown, no divergence); sensible learning-rate/optimizer choices for QAT; evidence of debugging effort (e.g. progressive quantization, warmup). |
| Evaluation & compression analysis | 5 | Complete comparison table (teacher vs. FP32 ResNet18 baseline vs. ternary KD student); correct model-size and sparsity computation; compression ratio computed and interpreted correctly; honest discussion of accuracy loss. |
| Report quality, ablation & critical discussion | 6 | Clear writing, correct citations of required papers; at least one meaningful ablation (T, lambda, or Delta); thoughtful discussion of limitations (e.g. real hardware speedup requires specialized kernels; ternary constraint impact on early/late layers). |

## Deliverable

10–14 pages of body plus appendices, as `report/main.tex` compiled to `report/report.pdf`.
Figures in `report/figures/`, generated by `report/make_figures.py` so they are reproducible.
`make_figures.py` reads only the CSVs and JSON in `report/data/` — matplotlib is available
locally and needs no torch, so figures can be iterated on the authoring machine.

There is no local LaTeX or pandoc. Either install MiKTeX locally, or compile on Colab with
`texlive-latex-recommended`. Ask before assuming which.

---

# Report structure

## 1. Introduction (1/2 page)

What is being compressed and why. State the claim under test exactly as section 1 of the
pre-registration words it: we measure the accuracy effect of ternarization **and how
distillation modifies it, in either direction**. Do not frame this as "recovering lost
accuracy" — no loss was presupposed.

State that the study was pre-registered before any training, that the frozen document is
Appendix A, and that all interpretation rules were fixed in advance. Flag this early rather
than letting the marker discover it.

### 1.1 Amendments (required)

The pre-registration carries an append-only amendment log. Summarise each one in the body
with its reason, and reproduce the log in Appendix A.1. Handled openly this is evidence the
process worked, not an admission:

- **A1** — section 6's arm E storage formula double-counted BatchNorm gamma and beta, which
  are already inside `parameters()`. Resolution: count each tensor once. Affects printed
  storage figures by well under 1% and **does not change arm E's selection**.
- **A2** — the handoff specified "2n BasicBlocks" per stage for the 6n+2 family. A block is
  two conv layers, so that yields 12n+2 layers and roughly double the parameters. Caught by
  cross-checking against He et al.'s published 0.27M for ResNet-20. Under the literal text,
  arm E would have been a substantially larger network.
- **A3** — resume checkpointing, so Colab preemption does not destroy runs. Section 10 lists
  preemption as a discard criterion and caps discards at 3 per arm, which the platform would
  have hit for reasons unrelated to the configuration under test.
- **A4** — which teacher checkpoint feeds arms B and D. The pre-registration is silent.
  Epoch 160 was used, recorded as `teacher_epoch` in every KD manifest: averaging the ten
  final checkpoints' weights is forbidden by section 12, and selecting among them by
  validation accuracy would be checkpoint selection. Epoch 160 is the only choice requiring
  no selection.

## 2. Ternary quantizer and student architecture (~2 pages) — rubric item 1

### 2.1 Formulation

Give the TWN problem: minimise ||W - alpha*W_tilde||^2 subject to W_tilde in {-1,0,+1}. Show
that for fixed Delta the optimal alpha is the mean magnitude of the surviving set, and that
this is implemented **per output filter** with n = the number of weights in that filter.

Note two source discrepancies, both verified against the paper text:

- TWN Eq. 6 prints `arg min` where substituting the optimal alpha yields a quantity that must
  be **maximised**. We implement the maximisation.
- We implement arXiv v3 (Nov 2022), whose Algorithm 1 is **per-filter with constant 0.75**.
  TTQ cites the earlier version as layer-wise with 0.7. State which we used.

### 2.2 Why per-filter, closed-form alpha

The alternatives and why they were rejected:

- **Learned scales (TTQ-style):** TTQ trains alpha with separate alpha_p and alpha_n and
  consequently cannot use the identity STE — it scales the backward gradient by alpha_p,
  alpha_n or 1 by region, an implicit time-varying per-region learning-rate multiplier that
  entangles scale and latent-weight learning rates. The closed form keeps the backward pass a
  clean identity and removes that coupling.
- **Per-layer alpha:** per-filter is what TWN Algorithm 1 does, and it is free at inference
  because the scale folds into the following BatchNorm.
- **Degeneracy with BatchNorm gamma:** both alpha and gamma scale an output channel, so
  (alpha, gamma) has a degenerate direction. The closed form removes it, because alpha is not
  trained — it is recomputed from the latent weights every forward pass.
- **Fixed-sparsity thresholds:** TTQ tested threshold against sparsity targets and report the
  threshold works better because it lets per-layer sparsity vary. We did not re-test this.

### 2.3 Degenerate filters

Describe the fallback: if no weight in a filter exceeds Delta, the largest-magnitude weight is
forced to survive. Justify it — with alpha = 0 the filter contributes nothing forward,
receives no useful gradient through the STE, and its latent weights only shrink further under
weight decay, an absorbing state it cannot leave.

**Then give the analytical result, which is more interesting than the counts.** Since
Delta = 0.75 * mean|W| and max|W| >= mean|W| > 0.75 * mean|W| for any filter that is not
identically zero, the support is non-empty for every non-degenerate filter. The fallback can
therefore fire **only on a bit-exactly-zero filter** — precisely the absorbing state it exists
to rescue — and its operative function is preventing alpha = 0/0 = NaN from poisoning the
forward pass. Report the measured counts from `degenerate_curves.csv` and state that counts of
zero are the expected structural outcome, not a logging failure.

### 2.4 Latent weights and the STE

**Figure 1 (architecture diagram).** The ternary block: latent fp32 weight -> quantise
(Delta, alpha) -> ternary weight -> convolution, with the backward path drawn as a **separate
arrow bypassing the quantizer** straight to the latent weight. Label forward and backward
distinctly. This figure carries most of the "latent FP weights maintained separately" and "STE
correctly implemented" criteria — make it clear and put it early.

Explain why latent weights are necessary: discrete weights cannot accumulate small SGD
updates, so the optimizer updates a real-valued master copy while the forward pass uses its
quantization. Cite Bengio et al. (2013) for the STE.

Justify the STE choice. Yin et al. give the criterion that a good STE matches the quantized
function at the extrema, and show the identity STE can be repelled from good minima. **State
the scope limit plainly:** their analysis is a two-linear-layer network with binarized
*activations* and float weights, so their convergence theorem does not transfer to weight
quantization. We use it as a design criterion, not a guarantee.

Explain the clamping choice: latent weights are clamped to [-1,1] after each step, with a
**vanilla identity** STE. Clamping was deliberately not combined with a clipped STE, because
that double-applies saturation — a weight at the boundary receives zero gradient there and can
never return, producing a growing population of permanently frozen weights that looks like
convergence.

### 2.5 Which layers stay FP32

Table: first conv, final FC, all BatchNorm parameters and buffers, with parameter counts and
percentage of total (from `summaries.json`). The reference papers disagree here — TTQ excludes
first/last on ImageNet but ternarizes the classifier on CIFAR; TWN states no exception — so
this is our documented choice, not an inherited convention. State the count of ternarized
convolutions and that it includes the 1x1 downsample convolutions in shortcut paths. These
counts feed section 6's arithmetic.

Note the architecture: ResNet-18 is the ImageNet architecture from He et al.; the CIFAR
adaptation (3x3 stride-1 stem, no maxpool) is standard practice and not from that paper. He
et al.'s own CIFAR models are the 6n+2 family at 16/32/64 channels, a different network — and
the family arm E is drawn from.

## 3. Knowledge distillation formulation (~1.5 pages) — rubric item 2

### 3.1 The loss

L = (1 - lambda) * CE(z_s, y) + lambda * T^2 * KL( softmax(z_t/T) || softmax(z_s/T) )

with lambda = 0.9, T = 4.

Explain the T^2 factor: soft-target gradients scale as 1/T^2, so Hinton et al. multiply by
T^2 to keep the relative contribution of hard and soft targets roughly unchanged as T varies.
It sits **inside** the soft term, before lambda, so lambda means the same thing at any
temperature. The effective soft-to-hard gradient ratio is lambda/(1-lambda) = 9:1.

State the parameterisation explicitly, or a reader comparing to Hinton will be misled: ours is
the convex form (1-lambda, lambda); Hinton's speech experiments used a relative weight of 0.5
on the hard-target cross-entropy against an unscaled soft term, a 2:1 ratio in a different
parameterisation. lambda = 0.9 is the later-standard convention, **not** Hinton's own value,
and was not tuned.

Two implementation details: `reduction='batchmean'` rather than `'mean'`, which would
additionally divide by the class count and rescale the soft term by 10x on CIFAR-10; and KL
rather than cross-entropy, which differ by a teacher-only constant that does not affect
gradients but does affect logged loss values.

### 3.2 Teacher freezing

**Figure 2 (architecture diagram).** Teacher and student side by side, the same augmented
batch into both, two loss paths into the combined objective, teacher box annotated `eval()`
and `no_grad()`. This figure is the "teacher correctly frozen, no gradient leakage" criterion.

Say why teacher logits are computed online rather than precomputed: augmentation is
stochastic, so cached logits would correspond to different inputs than the student sees. Say
why the teacher runs in `eval()`: BatchNorm using batch statistics would make the soft targets
depend on batch composition.

State which teacher checkpoint was used and why (amendment A4).

### 3.3 Why T and lambda were fixed rather than tuned

State this as a design choice with a reason, not a rule being obeyed.

Tuning T for the KD arms while the CE controls had nothing equivalent would compare a tuned
treatment against untuned controls. T and lambda were therefore fixed a priori at conventional
values, and T swept **separately** as a reported finding that could not and did not revise any
headline number. State that this biases against arm D — if T = 4 is not optimal for a ternary
student, D is reported below its potential — and that a bias running against our own claim is
acceptable where one running for it would not be.

lambda was not tuned at all. That is a budget limitation, recorded in section 8.

## 4. Experimental design (~1.5 pages)

No rubric line matches this section directly, which makes it pure upside: it is what makes
every number that follows credible.

### 4.1 Arms

The arms table (T, A, B, C, D, E) with seed counts. For each, the **specific alternative
explanation it rules out**:

- **T (teacher):** that a weak student result reflects a weak teacher.
- **A (fp32 CE):** the full-precision denominator. Rules out nothing alone; defines what
  "full precision accuracy" means for this architecture, split, augmentation and schedule.
  Published numbers are not a substitute.
- **C (ternary CE):** that KD did the work. A - C isolates ternarization's cost under
  identical supervision.
- **B (fp32 KD):** that any KD benefit is generic rather than specific to ternary students.
  Without B the interaction (D-C) - (B-A) cannot be computed.
- **E (nearest-storage fp32):** that a smaller float network would have achieved the same.
  **Describe E as nearest-storage, not storage-matched**, and give the residual: ResNet-44 is
  the closest of five candidates and sits below the ternary ResNet-18 target by the percentage
  in `summaries.json` — roughly 9%, not 0%. It was the only candidate inside the 5% tie-break
  band, so the tie-break never fired.

What was **not** run and why: label-smoothing control (would land inside noise at one seed),
full-budget warm-start control (obviated by training from scratch), feature-level KD arms
(would confound the interaction term), matched-FLOPs control (arm A serves that role, since
weight ternarization leaves MAC count unchanged).

### 4.2 Reporting protocol and why it is unusual

- **No checkpoint selection.** Best-epoch-across-training is a maximum over N draws and its
  upward bias scales with an arm's variance. The ternary arms are noisier by construction, so
  best-epoch reporting would inflate exactly the arms whose success we are claiming. We report
  the **mean of the final 10 epochs**, which is selection-free and does not reward variance.
  TTQ report a moving average over all epochs for the same reason without specifying a window;
  we specify ours.
- **45k/5k split.** Validation is used for learning curves, the settledness check and the
  teacher smoke test — never for selection or tuning. The test set was evaluated once, after
  all training for an arm completed.
- **The teacher trains on the same 45k**, so "no model saw the validation split during
  training" is unconditional.
- **From scratch, not warm-started.** Warm starting would give the ternary arms extra epochs
  the fp arms did not receive and make the ternary structure at step 0 a deterministic function
  of arm A. It also creates a specific failure mode: the latent weights never cross their
  thresholds, the discrete structure freezes at initialization, and what trains is only
  BatchNorm and the scales. From scratch, that mode does not exist.
- **Fixed budget, identical across arms.** N = 160, milestones at 80 and 120, inside the range
  used by TTQ and Yin et al. on CIFAR ResNets. We do **not** attribute the schedule to either:
  TTQ's printed milestones include epoch 300 alongside a claim of convergence at 160, which is
  internally inconsistent.
- **The split is verified, not assumed.** The 45k/5k partition is hashed and checked against a
  committed constant on every call, so drift is a hard error rather than a silent change in
  what "validation" means.

### 4.3 Pre-committed interpretation rules

State these as they appear in the pre-registration, and state that they were written before
any result was seen. They are implemented as code in `scripts/aggregate.py`, which emits the
verdicts — report what it computed, not what you concluded.

- **R = (D-C)/(A-C) is undefined** if C >= A or if |A-C| is smaller than the largest
  within-arm seed spread. Then raw accuracies are reported with no recovery language. The rule
  exists because a near-zero denominator would otherwise produce a spectacular-looking ratio.
- **The interaction (D-C) - (B-A)** is suggestive only if it exceeds twice the largest
  within-arm spread, and never a quantified effect size, because B has two seeds.
- **If T's test accuracy is at or below arm A's mean**, B and D are uninterpretable as
  distillation and the interaction is not computed.
- Any difference smaller than the largest relevant within-arm spread is **not resolved**, not
  a small effect.
- Spread means max - min throughout. Arm B's spread comes from two seeds; note that wherever
  it appears, including where it is the maximum.

## 5. Training pipeline and stability (~3 pages) — rubric item 3, the largest block

The rubric's examples of "debugging effort" are progressive quantization and warmup. The
pre-registration forbids both. **Do not let that read as an absence** — give the reason, then
present the verification harness as the debugging effort, because that is what it is:

> A schedule trick that helps one arm more than another shifts the comparison in a direction
> we cannot bound. Stability was addressed by measurement and pre-committed criteria rather
> than by interventions that would have confounded the arms.

### 5.1 The QAT loop

Per-step order: quantize -> forward -> backward (identity STE) -> optimizer step on latent
weights -> clamp to [-1,1]. Delta and alpha are recomputed **every forward pass**, not cached
per epoch, so the ternary forward is used consistently throughout KD training. Optimizer, LR
schedule, batch size, weight decay, and why weight decay is excluded from BatchNorm parameters
and biases.

Note the LR schedule is computed directly from the 1-indexed epoch rather than via
`MultiStepLR`, which with milestones (80, 120) and one `step()` per epoch drops at epochs 81
and 121 — an off-by-one that would shift every number.

Note weight decay's role under a relative threshold: because Delta = 0.75 * E|W| scales with
the weights, decay shrinks weights and threshold together and is close to inert on the ternary
assignments themselves. It was kept at the conventional value rather than removed, and not
tuned per arm.

### 5.2 Oscillation and the reversal-rate instrument

The centrepiece of this section.

Mechanism: a latent weight near a threshold can flip its ternary assignment on a tiny update,
flip back on the next, and keep doing so. The network used for computation changes even when
the latent weights barely move. Consequences are training noise and BatchNorm statistics that
describe an average over several distinct networks rather than one.

Define the metric precisely as implemented: a reversal is a state change whose direction
opposes that weight's **most recent prior change, however many epochs earlier**; the
denominator is all ternarized weights in the layer; it is undefined for epochs 1 and 2 and
logged as JSON `null`, because a 0.0 there would be indistinguishable from a stable epoch.

Distinguish it from a naive flip rate and say why that matters: a raw flip rate cannot separate
productive structural search from a small population of weights thrashing at boundaries, and
those call for opposite responses.

**Figure 3:** per-layer reversal rate vs epoch for arms C and D, from
`reversal_curves_per_layer.csv`. Discuss whether D is noisier than C — if distillation
destabilizes ternary training, that is a finding worth reporting in its own right.

### 5.3 Sparsity evolution

**Figure 4:** per-layer zero fraction vs epoch, from `zero_fraction_curves.csv`. Compare
qualitatively to TTQ, which reports minimum error at 30–50% sparsity, and to their observation
that early and late layers evolve in opposite directions — the first quantized conv losing
sparsity while the last conv and classifier gain it. State whether we reproduce that pattern.
This feeds the "ternary constraint impact on early/late layers" limitation.

### 5.4 Loss and accuracy curves

**Figure 5:** training loss and **validation** accuracy vs epoch, teacher and all student
arms, all seeds, from `metrics.jsonl` and `val_curves.csv`.

**Training accuracy was not logged** — the section 11 schema fixes the per-epoch fields and
does not include it. Say so in one line rather than omitting a curve silently, and do not
reconstruct or estimate it. The teacher's final training loss already shows the memorisation
behaviour that a training-accuracy curve would.

No divergence. Flatness over the final third is the evidence that the arms are not
undertrained — say so explicitly, since a reviewer's first objection to a fixed budget across
precisions is that the ternary arms converge more slowly and were cut short.

### 5.5 Stability criteria and discards

Degenerate filter counts per layer (see 2.3 for why these are structurally zero).

The settledness criterion (final-10 validation spread above 0.5 pp) and which runs were
flagged. **State that the criterion is weak and why**: it catches runs whose final-phase
variance makes a single number unreliable, and nothing finer. An earlier two-part version
comparing the final window against the pre-LR-drop window was considered and rejected as
near-vacuous, because a run noisy at the end is generally noisy earlier.

The teacher gate: its last-10 mean **validation** accuracy against the 92% threshold, from
`teacher_gate.json`. State that this is a pipeline smoke test, not a quality bar and not a
comparison against any student arm — the T-vs-A comparison is section 6's, on test, under the
pre-committed rule.

The discard rule, its three mechanical criteria, and per-arm counts from `discards.json`.
Emphasise that no criterion reads accuracy — a run that trains cleanly to a poor number is a
result, not a failure — and that replacement seeds came from a pre-listed sequence.

### 5.6 Verification harness and the defects it caught (required)

This subsection is where the "evidence of debugging effort" criterion is actually earned.
Describe the pre-flight suite: unit tests for the quantizer, STE, KD loss and reversal-rate
logic against hand-computed expectations; and integration checks for determinism, LR
boundaries, the ternarized layer set, the weight-decay split, data hygiene, restore-and-
reproduce, and resume fidelity. Nothing trains until they pass, enforced in code.

Then report the defects they caught, with what each would have cost. All are in the git
history:

- **The 6n+2 block count** (amendment A2) — caught by cross-checking a computed parameter
  count against He et al.'s published 0.27M for ResNet-20. Arm E would otherwise have been a
  roughly 2x larger network than intended.
- **Tracker state aliasing** — a restored tracker shared buffers with the original, so the
  original then observed no state change and reported zero reversals. Caught by a
  save/restore round-trip test.
- **An RNG-state device bug** — the resume loader mapped every tensor to the GPU, including
  the RNG ByteTensor, which the setter requires on CPU. Caught as a crash on the first real
  resume.
- **Sampler-generator divergence, the most instructive one** — resume ran without error but
  trained on the wrong data order. The shuffling generator advances one draw per epoch, so a
  freshly seeded loader handed epoch N the permutation epoch 1 should have had. Model,
  optimizer and RNG state were all restored correctly; the sampler's position in its own
  stream was not. It produced a small, entirely plausible difference in epoch-3 loss and
  validation accuracy. Report the two numbers from the verification log. Only a differential
  check — resumed run against uninterrupted run — could find it, and it had gone unnoticed
  until that check existed.

Draw the conclusion explicitly: a silent numerical divergence is worse than a crash, and the
checks that matter are the ones comparing two paths that must agree rather than asserting a
single path looks reasonable.

Also report the determinism evidence, which exceeds what the pre-registration required: the
same seed reproduced identical per-epoch loss and validation accuracy **across different
sessions on different cloud VMs**, a different seed diverged, and a reloaded checkpoint
reproduced its logged validation accuracy exactly. Give the actual numbers from the logs.

Note the mitigations adopted after these findings: checkpoints written via temp file plus
atomic rename so a killed runtime cannot truncate them, and every GPU model a run trained on
recorded in its manifest, since a run resumed on different hardware has a weaker bitwise
guarantee.

## 6. Evaluation and compression (~2 pages) — rubric item 4

### 6.1 Results table

Arm, seeds, last-10 test mean, spread, epoch-160 test, settledness flag, discard count. The
rubric wants **teacher vs fp32 ResNet-18 vs ternary KD student** — make T, A and D visually
prominent even though B, C and E are also present.

Apply the pre-committed rules and report the verdicts **as `aggregate.py` computed them**. If
R is undefined, say so and say which condition triggered it.

### 6.2 Honest discussion of accuracy

If the ternary arms match or exceed full precision, **say so directly and explain the
mechanism** rather than searching for a loss to report. ResNet-18 has roughly 11.2M parameters
on a 50k-image dataset; TTQ found ternarization improving on full precision for their deeper
CIFAR ResNets (32, 44, 56) while costing accuracy only on the smallest, ResNet-20, and
conjectured that ternary weights supply the right capacity and prevent overfitting. TWN report
the same direction: a wider backbone shrank the ternary-to-full-precision gap. A well-argued
"no loss, and here is why" beats a manufactured recovery figure.

If there is a loss, report it with the pre-committed R and do not round favourably.

### 6.3 Compression

Per-arm table: parameters, bits/weight, exported storage MB, MACs.

Show the arithmetic. State the fixed-width 2-bit convention explicitly, and that the total
includes per-filter fp32 scale factors, the fp32 first conv and final FC, and BatchNorm gamma,
beta, running mean and running variance — BatchNorm is shipped separately, not folded.
**Compute the ratio from `summaries.json`; do not cite 16x.** Both TWN and TTQ claim roughly
16x while keeping layers in fp32 and without counting scale storage; our figure is lower and
the difference is the point.

State that entropy coding or base-3 packing would reduce storage further, that we did not
compute a packed size, and why: it would make storage depend on trained weights and
retroactively change arm E's selection, which was fixed before training.

Per-layer sparsity against TTQ's 30–50% band.

**Interpretation.** Lead with matched storage, because that is the claim weight quantization
makes. In the same paragraph: MACs are unchanged, arm A is the matched-FLOPs comparison, and
no latency claim follows without custom kernels we did not write. Note that TTQ claim forward
time under 30% of full precision in the same section as their memory claim, without kernel
details, and that we do not inherit that claim.

Discuss arm E: if a nearest-storage float network lands in the same neighbourhood as D, that
is a substantive negative result about the whole pipeline and should be reported as one.

## 7. Ablation: distillation temperature (~1.5 pages) — rubric items 2 and 5

This section carries marks in two rubric rows. Numbered section, its own figure, not an
appendix.

**The hypothesis.** A ternary student has a smaller effective hypothesis space, and asking it
to match a high-entropy target may add noise on top of an already biased STE gradient rather
than transferring structure. Hinton et al. observed the width analogue on MNIST: with 300+
units per layer all temperatures above 8 behaved similarly, but at 30 units per layer
temperatures between 2.5 and 4 worked significantly better than higher or lower. Their section
2.1 gives the mechanism — at lower temperatures distillation pays much less attention to logits
far below average — and they conclude that when the student is too small to capture the
teacher's knowledge, intermediate temperatures work best.

**State the scope limit immediately.** That result varies *width* on MNIST MLPs. We vary
*precision* on a CIFAR ResNet. Whether reduced precision constrains the hypothesis space the
way reduced width does is an assumption we are testing, not a result we inherit. Hinton's
finding is "intermediate," not "low": the prediction is a moderate optimum, not a preference
for hard targets.

**Design.** Arm D at T in {2, 4, 8}, two seeds per point, everything else identical. Arm C is
the no-teacher reference line. **The T = 4 point is arm D at seeds 0 and 1 only** — not arm D's
three-seed mean — so all three points rest on the same seed count. `aggregate.py` builds it
that way; state it, because mixing a 3-seed mean against two 2-seed points would bias the
comparison.

**Ordering.** The headline arms were completed and their numbers recorded before any sweep run
began, enforced in code: the sweep phase refuses to start until every headline arm has its
test results written. If a temperature outperforms T = 4, report it as an unexploited finding.

**Claim scope.** This characterizes the ternary student's response to temperature. It is
**not** a test of the capacity-gap hypothesis, because no matched full-precision KD arm was
swept — that needs B and D over the same grid at two or more seeds, which we did not run for
budget reasons. Say so plainly; the honest smaller claim is worth more than an overreach.

**Reporting.** Monotone trends, or differences exceeding across-seed spread. Everything else
is flat. Report the verdict `aggregate.py` emitted. Two seeds per point locate a direction,
not an optimum.

**Figure 6:** test accuracy vs T for D, with per-point spread, and C as a horizontal reference
line.

## 8. Discussion and limitations (~1.5 pages) — rubric item 5

The rubric names two limitations. Cover both explicitly and prominently.

- **Real hardware speedup requires specialized kernels.** Weight ternarization leaves MAC
  count unchanged; TWN state that multiply-accumulates are unchanged relative to binary and
  the saving is that multiplications become additions. Ours is a storage result. No latency
  claim is made.
- **Ternary constraint impact on early and late layers.** The first-conv and final-FC
  exclusions, what they cost in parameters, and TTQ's finding that early and late layers evolve
  in opposite directions. Discuss the coupling between classifier precision and usable
  temperature: a ternary final layer computes a +/-alpha weighted sum over 512 features and
  cannot express finely softened targets, which is why FC precision and temperature should be
  considered jointly.

Then the rest, from the pre-registration's limitations section:

- N = 160 is compute-bounded; if the ternary arms are undertrained the bias runs against the
  claim, and the flat curves in 5.4 are the evidence offered against it. A 2x budget control
  was not run.
- T = 4 may not be optimal for a ternary student, biasing against D. lambda was not swept.
- B at two seeds limits the interaction to a directional statement.
- E at one seed supports a threshold judgment only, and is nearest-storage rather than exactly
  matched.
- Weight decay was not tuned per arm and may not suit ternary training.
- The 45k/5k split costs 10% of training data relative to published setups and may not affect
  all arms equally; arms with more effective capacity are expected to lose more.
- The degenerate-filter fallback overrides TWN's support selection, though it preserves the
  L2-optimal alpha for that support — and, per 2.3, can only fire on an all-zero filter.
- The settledness (0.5 pp) and teacher-gate (92%) thresholds are declared round numbers, not
  calibrated.
- Three seeds is the minimum that yields a spread at all; no formal statistical testing is
  performed and none is claimed.
- Runs executed on a shared cloud platform across many sessions. Resume fidelity is verified
  within a GPU model; a run resumed on different hardware has a weaker bitwise guarantee, and
  each manifest records every GPU it touched.
- Our absolute numbers are not comparable to any published CIFAR figure: different
  architecture (ResNet-18 vs the 6n+2 family), different training data (45k vs 50k), different
  reporting metric. Reference papers were used for direction and mechanism only.

Close with what a follow-up would do: B and D swept over the same temperature grid at higher
seed count; feature-level distillation as its own arms; a lambda sweep; a 2x budget control.

## 9. Conclusion (1/4 page)

What was found, in the pre-registered language. No claim stronger than the interpretation
rules permit.

## Appendices

- **A.** Full pre-registration verbatim. **A.1** the amendment log.
- **B.** Complete hyperparameter table, from a manifest rather than retyped.
- **C.** Per-layer statistics: zero fraction, reversal rate, degenerate counts, parameter
  counts.
- **D.** Logging schema and reproduction instructions. Include the **verifiable ordering
  evidence**: the pre-registration commit predates every run, and each manifest records its own
  `git_commit`, `gpu_model`, `torch_version` and seeds, so a reader can check the
  "pre-registered before training" claim rather than taking it on trust.
- **References.**

---

# Writing rules

**Every number comes from `report/data/`.** If you cannot find a number, stop and say so.
Never estimate, never fill in a plausible value, never carry a number from the reference
papers into a sentence about our results. Where `aggregate.py` emitted a verdict, report that
verdict rather than forming your own.

**State reasons, not rules.** Every pre-registered constraint appears with its justification in
the same sentence. Write "we fixed T a priori so the KD arms were not tuned against controls
that were not" — never "T was fixed per section 5 of the pre-registration." Rigor described as
obedience reads as evasion; rigor described with its reason reads as judgment.

**Be precise about citation scope.** Several papers are cited for things adjacent to what they
show. Flag the gap every time: Yin et al. analyse activation quantization with float weights;
Hinton's small-student result varies width on MNIST MLPs; He et al.'s CIFAR models are the
6n+2 family, not ResNet-18; TWN's CIFAR model is a VGG-7 variant. A marker who knows these
papers will notice, and noting it first is worth more than hoping they do not.

**Do not oversell.** If a result is inside noise, it is unresolved. If R is undefined, say so
and explain the rule. If arm E matches arm D, report it as a negative result about the
pipeline.

**Figures:** six in the body — two architecture diagrams, reversal rate, zero fraction,
loss/validation-accuracy curves, temperature sweep. Every figure has a caption stating what it
shows and what to conclude. Every figure is referenced in the text. All are produced by
`report/make_figures.py` from `report/data/`.

**Tone:** technical, direct, no filler. Short sentences. Avoid "notably," "it is worth
noting," "importantly." No em-dash asides, no "not X, but Y" constructions.

**Length:** 10–14 pages of body. If running long, cut from 1 and 9 first, never from 5 or 7.

---

# Before you write

1. Read `PREREGISTRATION.md` including the amendment log, and `ATDL_ASSIGNMENT1 Handoff.md`.
2. Confirm `report/data/` is populated and read `report.md` and `summaries.json`.
3. Read `src/quantizer.py`, `src/losses.py`, `src/tracking.py`, `src/train.py`.
4. Write `report/make_figures.py`, generate all six figures, and look at them.
5. Then draft the report.

If the data contradicts anything in this prompt — for example if the ternary arms lose accuracy
where this prompt allows they might not — **follow the data and the pre-registered rules, not
this prompt's expectations.** Report any contradiction you find rather than resolving it
silently.
