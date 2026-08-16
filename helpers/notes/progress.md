# CLC segmentation — progress log

Focus: improving Cellpose-SAM cell-mask segmentation on the in-house **CLC**
cochlear dataset (20 animals / 67 images, ~140 cells each), and an error-driven
investigation of where the model fails. Companion docs:
`notes/clc_seg_experiments.md` (plan), `notes/clc_seg_results.md` (full results),
`notes/RUNBOOK.md` (exact commands), `notes/model_train_log.md` (per-model
provenance: config + job + command for every trained artifact).

Last updated: 2026-07-31 (added the backbone comparison — it finished 2026-06-23
after this doc was last written — and the 2026-07 +20-image data re-test).

---

## TL;DR — current status

- **DEPLOYED (prod):** `clc_all_full_deploy` — the winning recipe trained on all
  67 images with no held-out fold (job 2702929, `trainer_deploy.yaml`). Training
  code + exact run commands: `notes/model_train_log.md` §1 / `RUNBOOK.md` §4a.
- **Its honest estimate** is the CV twin `clc_all_full_20260604` — same recipe,
  same config, 4/5 of the data per fold: single `clc_all` fine-tuned from cpsam
  on the **full 67-image set**, grouped 5-fold age-stratified CV, AP@0.5
  **0.820** (vs 0.834 on the easier early 15-image subset — the full set is
  genuinely harder). One model for both ages.
- **Backbone is not a lever:** cpdino / cpdino@384 / cpsam_v2 all statistically
  tied with cpsam (see "Backbone comparison" below). cpsam stays.
- **The data lever has saturated — tested twice.** 2026-07: +20 images (87 total)
  changed nothing (AP@0.5 +0.004, p=0.90; FN-recovery +0.1 pt), nor did excluding
  the batch's out-of-domain 20x image (p=0.89). 2026-08: +12 more images from a
  new adult prep plus 6 label corrections (98 total) — again nothing (AP@0.5
  +0.009, p=0.96; FN-recovery +0.4 pt). Prod model unchanged both times. See
  `model_train_log.md` §4a/§4b.
- **NEW open failure mode — severe hair-cell degeneration.** On cochleae with
  near-total hair-cell loss (4–13 cells/image, the 2026-08 `tomt overexp` prep)
  the deployed model scores **0.177** mean AP@0.5 vs 0.828 on everything else —
  it predicts *zero* masks on a 4-cell image, or over-detects on damaged tissue.
  Training on two such animals did **not** fix it. Needs its own intervention,
  not another general retrain. `model_train_log.md` §4b.
- **GT provenance (verified):** model-assisted manual annotation — cpsam
  proposed masks, a human curated every image (removed false masks, hand-drew
  ~6–14% missed cells). ~89% of mask boundaries are cpsam's full-cell geometry.
- **FP status: SOLVED.** Fine-tuning suppresses **~90%** of the false positives
  cpsam produced. No FP-rejection post-processor needed.
- **FN challenge: IMPROVING.** On the early 15-image subset fine-tuning
  recovered only ~56% of missed cells, and threshold tuning did **not** help
  (it's under-segmentation/merging of large elongated OHCs). **Rebuilding on the
  full 67-image set lifted FN-recovery to 71.9% (+16 pts)** — same config, pure
  data effect — with FP-suppression held (89.6%) and TP-retention up (95.2%).
  More representative data was the biggest single lever; residual 28% FN remains.

---

## Dataset overview

All CLC images: single-channel uint8, 1024×1024, ~130–190 cells each; labeled in
the Cellpose GUI (model-assisted: cpsam proposed, human curated). Group key for
leak-free splits = **animal** (multiple tonotopic frequency regions 8/16/32 kHz
share a cochlea). Source: `/data/cellpose_cc/{adult,neonate}/` (local).

| | Full labeled set | CV / training set used so far |
|---|---|---|
| images | **67** | **15** (~22%) |
| adult | 32 imgs / **12 animals** | 7 imgs / **4 animals** (5042L, 5044L, 5165, 5168) |
| neonate | 35 imgs / **8 animals** | 8 imgs / **4 animals** (1L, 2L, 3L, 4L) |
| selection | all labeled | early batch = samples #1 + #10–16 (was offline-D4-augmented) |

- **The CV was NOT adult-only** — 4 adult + 4 neonate animals (both ages).
- Full adult animals: 5042L, 5044L, 5048L, 5049L, 5056L, 5058L, 5165, 5167,
  5168, 5457, 5457L, 5460L. Full neonate: 1L–8L.
- **Early-June additions** (the lab kept labeling): +8 adult animals, +4 neonate
  animals — and these new images are **FN-rich** (many high-freq 16/32 kHz
  regions with FN ≈ 90–130, e.g. 5049L, 5056L, 5168-32k, 7L, 5L, 6L). They are
  exactly the hard, densely-packed-OHC cases the model currently fails on.
- So the expanded set is not just bigger — it is concentrated in the failure
  mode, which makes it the key asset for attacking FN.

---

## The FP / FN picture (from the curation error-eval, Approach B)

The CLC curation log (`ismanual` + `manual_changes` in each seg) is a labeled
record of the base model's errors. Scored leak-free with the LOAO held-out
fine-tuned fold models via `clc_error_eval.py`.

Error sets across 15 images: **TP 1787** (kept), **FN 345** (hand-added =
cpsam-missed), **FP 842** (cpsam masks the curator removed).

| model | FN-recovery | FP-suppression | TP-retention |
|---|---|---|---|
| cpsam (the proposer) | 27.5% | 0.0% | 99.7% |
| **clc fine-tuned (held-out)** | **55.9%** | **91.0%** | 93.0% |

(cpsam's 0% FP-suppression / 99.7% TP-retention = sanity check: it reproduces
its own proposals.)

### FP status — solved
Fine-tuning learned the curator's removal decisions end-to-end: **91% of cpsam's
false positives are suppressed**, at a small cost (TP-retention 99.7% → 93%, the
model is slightly more conservative). A dedicated FP-rejection second model is
**not worth building**.

> **MEASURED 2026-08-15 (`model_train_log.md` §4d): the residual FN is 56%
> merging, 44% missed outright.** The "it's under-segmentation" framing below
> was inferred from morphology, threshold behaviour and figures, never measured.
> Direct measurement (`helpers/probes/fn_partition.py`, 1239 residual FN cells): 694
> have their centroid inside a predicted mask (merged), 545 land in background
> with nothing predicted at all. Splitting-based fixes have a hard ceiling of
> 86.5% FN-recovery; the other half needs detection-side work.

### FN challenge — open, and it's under-segmentation, not low confidence
The cells the model still misses are **large, bright, elongated** (median area
~2× a typical cell, eccentricity 0.95) — the big oblong OHCs — and are
concentrated in a few hard images (`5168`: 91 FN; `1L`-32khz: 56; `5165`-16khz:
37). Fine-tuning recovers ~56% of FN but plateaus there.

**Over-detect + filter was tested and refuted** (`clc_threshold_sweep.py`,
sweep cellprob{0,−1,−2,−4,−6} × flow{0.4,0.8}):

| flow | cellprob | FN-recov | TP-ret | modelFP/img | AP@0.5 |
|---|---|---|---|---|---|
| 0.4 | 0.0 (default) | **55.9%** | 93.0% | 8.1 | **0.827** |
| 0.4 | −2.0 | 51.9% | 91.7% | 8.8 | 0.806 |
| 0.4 | −6.0 | 42.6% | 86.8% | 7.9 | 0.757 |
| 0.8 | 0.0 | 57.4% | 93.4% | 10.5 | 0.821 |

The default threshold is already optimal; **lowering `cellprob_threshold` makes
everything worse** because densely-packed cells **merge** when masks expand. So
the missed cells are an **under-segmentation/merging** problem, not a
detection-confidence problem — no threshold or post-filter lever recovers them.

### Visual evidence (figures in this folder)
- `hard_images_FN.png` — the four hardest images, full frame, missed cells in
  **red**, kept cells faint cyan.
- `hardest_zoom_FN.png` — cell-level zoom of the hardest image.

What the figures show (confirms the merge/under-segmentation hypothesis):
- Missed cells are the **large, elongated outer hair cells packed tightly in
  vertical rows, nearly touching** — exactly the arrangement that drives
  merging. The inner-hair-cell row (smaller, rounder, single file) is also
  missed but is secondary.
- **The hardest images are near-fully manual** (`kept ≈ 0`): on whole **32 kHz**
  regions cpsam fails wholesale and the human hand-labeled almost everything. So
  the FN problem is worst in high-frequency regions, not uniform.

> **Dataset caveat — the CV set is only ~22% of the labeled data.** The
> 15-image cluster CV set is **samples #1 + #10–16** (adult `1,10–15`; neonate
> `1,10–16`) — the early batch that had been offline-D4-augmented for the first
> CLC experiments. `clc_split.py` used all originals present on the cluster; the
> 15 were never deliberately selected (incidentally animal-coherent because
> sample numbering follows animals; the LOAO folding is the real animal
> safeguard). The **full local set (`/data/cellpose_cc/`) is 67 images** — adult
> samples 1–32 (32 imgs), neonate 1–37 (35 imgs) — ~4.5× larger, with more
> animals and the hard 32 kHz regions (samples 17+) the subset under-represents.
> So the CV results (AP@0.5 0.834, FN-recovery 56%) likely **understate** real
> performance. **Next round should rebuild on the full 67-image set** (re-run
> `clc_split.py` over all of it, upload to cluster, redo LOAO CV).

---

## Adult vs neonate — was training them separately better? (Result 6)

**Yes, this was tested** (2026-06-02, jobs 2647416 `clc_adult` / 2647417
`clc_neonate`): age-specific models (trained on one age only) vs the single
mixed `clc_all`, all LOAO-CV from cpsam, same augmentation. The CV set had both
ages (4 adult + 4 neonate animals) — it was **not** adult-only.

Paired, on the same held-out images:

| subset | metric | age-specific | clc_all (mixed) | verdict |
|---|---|---|---|---|
| adult (n=7) | AP@0.5 | 0.799 | 0.804 | tied (p=0.47) |
| adult | AP@0.75 | 0.536 | 0.584 | clc_all better (p=0.078) |
| adult | AP@0.9 | 0.272 | 0.348 | clc_all better (p=0.078) |
| neonate (n=8) | AP@0.5 | 0.839 | 0.859 | **clc_all better, 7/8 (p=0.016)** |
| neonate | AP@0.75/0.9 | 0.725/0.443 | 0.721/0.465 | tied |

**Finding:** the single mixed model matched or beat age-specific on both ages →
at the time we concluded "deploy one mixed model."

> **CAVEAT — confounded by data starvation; RE-TEST on the full set.** The
> specialists trained on only **3–4 animals each** (the early 15-image subset),
> so "specialization loses" conflated *specialization* with *training-set size*
> (e.g. `clc_adult` fold 4 overfit: train loss 0.18 / test 0.55). The expanded
> set has **12 adult + 8 neonate animals**, so an adult-only model would now
> train on 3× more data. Since neonate vs adult hair cells are developmentally
> distinct (immature vs mature), age-specific models could genuinely win once
> properly fed — **the earlier "no" may flip.** The full-set rebuild should
> re-run the three-way comparison (`clc_all` vs `clc_adult` vs `clc_neonate`)
> with the specialists no longer data-starved. Same CV machinery, three recipes.

---

## Steps taken

1. **Built online augmentation** (`aug_online.py` + `train_seg(img_transform=)`
   hook) — photometric/cutout/noise, image-only, zero disk. Validated on
   Cunningham: parity with the deleted offline D4 files at matched
   `nimg_per_epoch` (which defaults to file count — must set explicitly).
2. **Built eval + CV infra:** `eval_seg.py` (AP@IoU), leak-free
   GroupKFold-by-**animal** split (`clc_split.py`), LOAO CV orchestration
   (`clc_cv.py` + `run_clc_cv.slurm` + `submit_cv.sh`).
3. **CLC baseline:** stock cpsam (0.648 AP@0.5) >> Cunningham-tuned model
   (0.411). The "bar" is cpsam, not the Cunningham model.
4. **Axis A (data composition), LOAO CV:** `clc_all` from cpsam **0.834** >
   cpsam (p=0.004); warm-start tied; per-age models lose. → one mixed model.
5. **Verified GT provenance** from seg metadata: model-assisted manual
   annotation (cpsam + human curation; ~89% cpsam boundaries, ~11% hand-drawn).
6. **Approach B — curation as error test-set** (`clc_error_eval.py`):
   FP solved (91% suppressed), FN half-recovered (56%).
7. **Threshold sweep** (`clc_threshold_sweep.py`): over-detect+filter refuted;
   FN is under-segmentation of large/elongated cells.
8. **Full-set rebuild (DONE):** staged all 67 images to `CLC_full/`, grouped
   5-fold age-stratified CV, retrained `clc_all` from cpsam (same config).
   **FN-recovery 56% → 71.9%** (pure data effect); FP-supp 89.6%, TP-ret 95.2%;
   CV AP@0.5 0.820. → more data is the biggest FN lever.

(Head-only mask question explored separately and **blocked** — heads are not
visually separable in the 8-bit CLC images; likely need 16-bit originals. See
`notes/clc_seg_results.md` §4½.)

---

## Completed experiments (full set) + next steps

**Step 1 — full-set rebuild (DONE):** FN-recovery 56% → 72% purely from more
data. Current best = `clc_all_full_20260604` (grouped 5-fold CV on 67 images,
AP@0.5 0.820).

**Three-way age comparison (DONE — mixed model confirmed):** re-ran Result 6 on
the full set (12 adult / 8 neonate animals, no data starvation). Paired,
matching-age held-out:
| age | model | FN-recovery | AP@0.5 |
|---|---|---|---|
| adult (n=32) | clc_adult | 79.2% | 0.828 |
| adult | clc_all (mixed) | 77.6% | 0.822 — **tied** (p=0.68) |
| neonate (n=35) | clc_neonate | 62.9% | 0.804 |
| neonate | clc_all (mixed) | **64.6%** | **0.819 — mixed sig better** (p=0.046) |
The earlier "no" did **not** flip → **deploy ONE mixed model**. Age asymmetry:
neonate harder (FN-recov ~63% vs adult ~78%) and **benefits from adult cross-age
data**; adult is self-sufficient (biologically sensible: mature vs immature HCs).

**FN-focused augmentation (DONE — no effect):** `clc_all_fnaug` (drop cutout +
gauss_blur, same data/config) → FN-recovery **72.1%** vs baseline 71.9%, all
metrics + AP flat. → augmentation *content* is NOT limiting FN; the residual
~28% FN is the fundamental dense-cell **merging** problem.

**Boundary/separation-aware loss (DONE — no effect):** flow-gradient boundary
weighting in `_loss_fn_seg` (`boundary_weight` α; α=0 = exact baseline),
`trainer_bndry.yaml` α=5. 5-fold CV: FN-recovery **71.1%** vs 71.9%, FP-supp
**identical** (89.6%), TP-ret 95.0%, AP@0.5 0.814 vs 0.820 — flat-to-slightly
worse, **not** over-splitting (FP unchanged). Reweighting boundaries 5× did
nothing → refutes the optimization-signal hypothesis; **merging is a
resolution/architecture limit, not a loss problem.**

**Backbone comparison (DONE 2026-06-23 — all tied):** does a different
pretrained backbone segment these cells better? Four arms, same 5 folds, same
recipe, paired per-image vs the `clc_all_full` control (AP@0.5 0.8203, n=67):

| arm | init / trainer | AP@0.5 | Δ vs control | p | FN-rec / FP-supp / TP-ret |
|---|---|---|---|---|---|
| cpdino (stock aug) | `cpdino`, upstream env | 0.8190 | −0.0013 | 0.85 | 71.2 / 89.7 / 95.1 |
| cpdino @384 crop, b4 | `cpdino`, `trainer_dino384.yaml` | 0.8177 | −0.0026 | 0.72 | — |
| cpsam_v2 (stock aug) | `cpsam_v2`, upstream env | 0.8086 | −0.0117 | 0.117 | 69.0 / 89.4 / 95.0 |
| cpsam_v2 + fork online aug | `cpsam_v2`, fork trainer | 0.8235 | +0.0032 | 0.348 | — |
| **clc_all_full (cpsam)** | control | **0.8203** | — | — | 71.9 / 89.6 / 95.2 |

→ **No backbone beats cpsam**; nothing is even close to significant, and the
crop-size lever (256 → cpdino's native 384) moved nothing either. Jobs 2702880 /
2702957 / 2702958 / 2703588, FN eval 2703245, pooling `finish_dino.sh` →
`run_logs/dino_summary.txt`. The dino/v2 scripts live **only on the cluster**
(`model_train_log.md` §4). **Keep cpsam.**

**2026-07 batch (+20 images) — DONE, no improvement, prod unchanged.** The lab
added 20 curated images (new animals `8363` adult, `AL/BL/CL/DL` neonate + 3
re-imaged `4L`) → `CLC_full2`, 87 images / 25 animals. Tested **paired** (fold
assignment pinned via the new `clc_split.py --extend`, control = the existing
`clc_all_full_20260604` fold models re-scored on the extended folds — harness
verified: old images reproduce at max|Δ| = 0):

A third arm repeated it with the batch's lone out-of-domain **20x** acquisition
dropped from training. All three scored on the same 86 non-20x images:

| metric | A: 67 (deployed) | B: 87 | C: 86 (no 20x) |
|---|---|---|---|
| AP@0.5 | 0.8165 | 0.8222 | 0.8194 |
| AP@0.5 (old 67) | 0.8203 | 0.8206 | 0.8193 |
| AP@0.5 (new 19) | 0.8032 | 0.8280 | 0.8200 |
| FN-recov / FP-supp / TP-ret | 69.1 / 89.4 / 95.4% | 69.2 / 89.3 / 95.8% | 69.3 / 89.5 / 95.6% |

Paired AP@0.5: B vs A +0.006 (p=0.94) · C vs A +0.003 (p=0.64) · **C vs B −0.003
(p=0.89)**.

→ **Diminishing returns confirmed**: 15→67 images (4.5×) bought +16 pts of
FN-recovery; 67→87 (1.3×) buys nothing, with or without the 20x image. No
regression on the old set either, so the new images stay in the dataset for
future refreshes. **The 20x image was a bystander, not a contaminant** — it is
the worst *test* image (0.592 → 0.427, it's off-domain) but removing it from
*training* changed nothing; 20x needs its own model, not data cleaning (cpsam
has no diameter-rescale step). Jobs 3346056 (B), 3346149 (C), 3346061 (control),
3346128 / 3346206 (error-evals); details in `model_train_log.md` §4a.

**2026-08 batch (98 images) — DONE, no improvement, prod unchanged.** 4 new adult
animals (`8477/8478/8480/8483`, a "myo 7e10 tomt overexp" prep) + 6 re-curated
labels → `CLC_full3`, 98 images / 29 animals (the 20x image excluded by decision:
20x is a separate training target). Same paired design (`--extend`, control
re-scored — **mandatory this time**, since 6 images' GT changed and the old CSVs
were stale):

| metric, n=98 paired | arm D (98) | control (67) | Δ | p |
|---|---|---|---|---|
| AP@0.5 | 0.7996 | 0.7907 | +0.0089 | **0.96** |
| AP@0.5 (old 67) | 0.8169 | 0.8210 | −0.0041 | 0.33 |
| AP@0.5 (92 non-sparse) | 0.8175 | 0.8169 | +0.0006 | 0.79 |
| FN-recov / FP-supp / TP-ret | 69.7 / 89.0 / 95.5% | 69.3 / 89.2 / 95.5% | ~0 | — |

The 2026-08 subset shows +0.082 and the 6 degeneration images +0.136, but both
are **small-denominator noise** — 3 of those 6 got worse, and the whole subset
gain is one image picking up two detections. Jobs 3479313 (arm D), 3479318
(control + deploy-on-unseen), 3479399 (error-eval). Two methodology lessons are
recorded in `model_train_log.md` §4b: **diff `masks`/`ismanual`/`manual_changes`,
never mtimes** (31 of 32 files were rewritten by the GUI but only 6 changed
labels), and **report GT-cell counts beside every subset** (per-image AP means
mix 4-cell and 238-cell images at equal weight).

**20x inclusion + a hold-out that can see the sparse case (2026-08-14) — both
negative, plus the noise floor.** Two follow-ups (`model_train_log.md` §4c):
- **arm E** (99 imgs, 20x included): no effect on the 63x set (+0.002 vs arm D,
  p=0.36; FN-rec 69.9%). The 20x image itself is **unusable by every model** —
  `clc_all_full_deploy` finds 205 of its 1262 cells (AP 0.126), the CV models
  0.03–0.10. → 20x needs its own model, not inclusion here.
- **arm F** (98 imgs, `--pin 8483=0` so each degeneration animal is tested by a
  model trained on the other): FN-rec **70.2%**, the best of any arm, but still
  tied (AP@0.5 +0.014 vs control, p=0.80). Degeneration transfer is unproven.
- **NOISE FLOOR (the important bit).** Arms D and E share identical fold-3
  training data, so D-vs-E on the 6 degeneration images is a same-data replicate:
  **mean |ΔAP| = 0.124, max 0.400** with no data change. One image moved
  0.800→0.400 on nothing but a seed. Over all 98 images the replicate Δ is only
  **+0.0022**. So: 98-image verdicts are well powered; **any claim resting on a
  handful of sparse images is noise unless a replicate says otherwise.**

### Conclusion — FN investigation complete; deploy `clc_all_full`
Every tractable FN lever tested: more data **+16 pts** (the only win, and now
**saturated** — a further +20 images did nothing) · threshold tuning (merges) ·
aug content (no effect) · age specialization (no benefit) · boundary loss (no
effect) · **backbone swap (no effect)**. The residual ~28% FN is a **fundamental
limit** of densely-packed touching cells at 8-bit / pixel resolution.

**Deliverable: `clc_all_full_deploy`** — the winning recipe (clc_all from cpsam,
full aug, nimg 750) trained on **all 67 images, no held-out fold** (job 2702929,
`trainer_deploy.yaml`, → `/ix1/pcody/cellpose/models/clc_all_full_deploy`). Its
expected performance is the leak-free CV estimate from `clc_all_full_20260604`:
FN-recovery **72%**, FP-suppression **90%**, AP@0.5 **0.82** on the full set;
beats stock cpsam decisively. Register with `python -m cellpose --add_model`.
FP work, age specialization, aug, threshold, and loss levers are all **done**.

Untested-but-unlikely: a precise instance-boundary weight (Variant B) or much
higher α — but α=5 moved nothing, so low expected value. Real headroom would need
higher-bit-depth source images (see `clc_seg_results.md` §4½) or super-resolution,
not another training knob.

Repo note: `cellpose/train.py` boundary-loss edit is kept (α=0 default = no-op);
`trainer_slurm.py` boundary_weight threading was reverted locally (the running
job used the deployed cluster copy) — re-apply only if boundary loss is revisited.

---

## Key artifacts (all in `helpers/`)

| file | role |
|---|---|
| `aug_online.py` | online image-only augmentation (`img_transform`) |
| `eval_seg.py` / `run_eval.slurm` | AP@IoU head-to-head eval |
| `clc_split.py` | leak-free GroupKFold-by-animal manifest |
| `clc_cv.py` / `run_clc_cv.slurm` / `submit_cv.sh` | LOAO CV train+eval |
| `clc_error_eval.py` / `run_erreval.slurm` | curation-as-error-set eval (B) |
| `clc_threshold_sweep.py` / `run_sweep.slurm` | over-detect threshold sweep |
| `trainer.yaml` / `trainer_warmstart.yaml` / `trainer_fnaug.yaml` / `trainer_bndry.yaml` / `trainer_deploy.yaml` | run configs (data/train/augment) — one per experiment arm |
| `probes/head_pilot.py` | head-separability feasibility probe (blocked) |
| `notes/RUNBOOK.md` | exact commands for every step (incl. §4a deploy-model run code) |
| `notes/model_train_log.md` | per-model provenance: config + data + job + command |
