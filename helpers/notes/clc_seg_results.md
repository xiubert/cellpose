# CLC segmentation experiments — results

Results log for the effort to improve Cellpose-SAM **cell-mask segmentation** on
the in-house **CLC** cochlear dataset. Companion to the plan/method doc
[`clc_seg_experiments.md`](clc_seg_experiments.md); this file is the findings.

**Metric:** average precision at IoU 0.5 / 0.75 / 0.9
(`cellpose.metrics.average_precision`) + mean predicted vs true cell counts.
AP@0.5 is the primary metric (correct cell detection — what matters for
hair-cell counting/classification); AP@0.9 reports boundary precision.

**Status (2026-06-23): CLOSED — `clc_all_full_deploy` shipped.** §1–§4 below are
the **early 15-image era**; the final full-set (67-image) numbers, the levers
that did *not* work, and the backbone comparison are in **§5**. Provenance for
every trained artifact: [`model_train_log.md`](model_train_log.md); commands:
[`RUNBOOK.md`](RUNBOOK.md); narrative status: [`progress.md`](progress.md).

> **⚠️ Reframing of 2026-06-03 (§4½) — RESOLVED AS BLOCKED, keep reading.** The
> head-only retarget below was investigated and **did not proceed**: the head is
> not visually separable in the 8-bit CLC images (see §4½ "Outcome"). The
> full-cell target therefore **stands**, and so do §3–§5's numbers.

> The CLC ground truth
> encloses the **full oblong cell**, but the desired target is the **hair-cell
> head** only: the downstream IHC/OHC classifier performs better on heads, and
> SAM segments heads more cleanly. The Cunningham label-transfer masks are
> already head-only. This means (a) the AP numbers below are scored against the
> *wrong* (full-cell) target, (b) the apparent "Cunningham model is bad on CLC"
> result (§3) is largely an **annotation-style artifact**, and (c) the deploy
> recommendation in §5 optimizes the wrong target until the CLC GT is converted
> to head-only. The CV *machinery* and the augmentation findings (§2) stand.

---

## 1. Datasets & method

| | CLC (target) | Cunningham / PMID-38653806 (label transfer) |
|---|---|---|
| Images | 15 originals | 93 train / 23 test originals |
| Cells/img | ~142 | ~62 |
| Groups | **8 animals** (4 adult: 5042L, 5165, 5168, 5044L; 4 neonate: 1L, 2L, 3L, 4L) | by source image |
| Offline D4 aug | deleted (online aug instead) | deleted |

- **CLC validation = leave-one-animal-out (LOAO) CV.** Group by **animal**, not
  image: multiple tonotopic frequency regions (8/16/32 kHz) come from one
  cochlea and must not straddle train/test. 8 folds; each holds out 1 animal
  (1–3 imgs), trains on the other 7 (12–14 imgs).
- **Aggregation:** pool per-image AP across all folds (every image held out once)
  → one CV estimate over 15 (or the age subset) images. Configs compared by a
  **paired** Wilcoxon signed-rank test on the same per-image APs.
- **Augmentation:** cellpose already does geometry (rotate/flip/scale/crop) on
  the fly; we add online photometric + occlusion (intensity/gamma, cutout,
  noise, blur) — image-only, zero disk (`aug_online.py`). `nimg_per_epoch=750`
  fixes the per-epoch crop budget (critical on originals-only data).
- **Caveat throughout:** 15 images / 8 animals is small — CIs are wide; don't
  over-read ~0.01 AP gaps.

---

## 2. Online augmentation validates on Cunningham (Results 1–2)

First we checked the new online aug against the old offline-D4 pipeline on the
Cunningham test set (23 imgs), since the offline `_SV_*` files were deleted.

| model | aug / budget | AP@0.5 | AP@0.75 | AP@0.9 | n_pred (true 62) |
|---|---|---|---|---|---|
| `label_xfer_aug_retest` | offline D4 (744 files), 95 ep | **0.808** | **0.601** | 0.305 | 58.7 |
| `label_xfer_online_aug` | online, `nimg`=default(93) | 0.735 | 0.570 | 0.238 | 53.2 |
| `label_xfer_online_aug_nimg750` | online, `nimg=750` | 0.787 | 0.595 | **0.304** | 57.8 |

**Finding:** the first online run lost only because `nimg_per_epoch` defaults to
the file count — with offline aug deleted it saw ~8× fewer crops/epoch
(undertrained). Budget-matched (`nimg=750` ≈ old 93×8), **online aug reaches
parity** with offline D4 (tied at AP@0.9; ~0.02 behind at AP@0.5, within noise).
→ Online aug is a sound zero-disk replacement. **Always set `nimg_per_epoch`
explicitly on originals-only data.**

Jobs: train 2644575 / 2644707; eval 2644683 / 2646554.

---

## 3. CLC baseline — the key reframing (Result 3)

Before any CLC training, scored the two existing models on all 15 CLC originals.

| model | AP@0.5 | AP@0.75 | AP@0.9 | n_pred (true 142) |
|---|---|---|---|---|
| **stock `cpsam`** | **0.648** | **0.526** | **0.510** | 181 |
| `label_xfer_aug_retest` (Cunningham "best") | 0.411 | 0.232 | 0.040 | 112 |

**Stock cpsam scores far higher than the Cunningham-tuned model against the
full-cell CLC GT.** Originally read as "the Cunningham fine-tune overfit its
domain and doesn't transfer."

> **Revised interpretation (2026-06-03):** this is largely an **annotation-style
> artifact**, not domain failure. `label_xfer_aug_retest` draws **head-only**
> masks (Cunningham style); the CLC GT encloses **full cells**. IoU(head,
> full-cell) ≈ head_area / cell_area is low *by construction* — moderate at
> AP@0.5, ~zero at AP@0.9. The observed 0.41 / 0.23 / **0.04** is exactly that
> signature. So the Cunningham model may be producing the *desired* (head) masks
> while being penalized against the *wrong* (full-cell) reference. **The "bar =
> stock cpsam" framing only holds for the full-cell target.** Once CLC GT is
> head-only, the Cunningham model is expected to score far better. See §4½.
>
> **cpsam's 0.648 is also partly inflated by the GT provenance** — see the
> provenance box below: ~89% of CLC mask boundaries are unedited cpsam output,
> so cpsam reproduces most cells exactly (the model that proposed them).

> **GT provenance — verified (2026-06-03), model-assisted manual annotation.**
> CLC seg metadata: `model_path`=cpsam; `ismanual` flags only **6–14%/image** as
> hand-drawn; `manual_changes` (69–169/image) is mostly *"removed mask"*. Per-cell
> IoU vs re-run cpsam on one adult image (132 cells), split by `ismanual`:
> **non-manual cells (89%) → median IoU 1.000, 86% exact pixel match; manual cells
> (11%) → median IoU 0.397, 0% exact**; cpsam proposed 155 masks, 23 curated out.
> So the workflow was: **run cpsam → a human curated every image** (removed wrong
> masks, hand-added ~11% missed cells, kept cpsam boundaries elsewhere). The
> labels are human-validated (real signal), but ~89% of mask *geometry* is cpsam's
> full-cell style. Consequences: cpsam's 0.648 and `clc_all`'s 0.834 are both
> **somewhat flattered** by the cpsam-derived target (not circular — the model
> must still learn the curation: suppress the false masks, add the hand-drawn
> cells). Earlier draft's "GT = cpsam output" was an overstatement.

Job 2646778.

---

## 4. CLC fine-tuning beats the bar (Results 4–6)

All CLC models: LOAO CV, trained from stock cpsam (except warm-start), online
aug, `nimg=750`. Pooled per-image AP.

### Master comparison

| model | train data | AP@0.5 | AP@0.75 | AP@0.9 | vs cpsam (AP@0.5) |
|---|---|---|---|---|---|
| stock cpsam | — | 0.648 | 0.526 | **0.510** | — |
| label_xfer_aug_retest | Cunningham | 0.411 | 0.232 | 0.040 | −0.237 |
| **clc_all (from cpsam)** | all CLC, both ages | **0.834** | 0.657 | 0.410 | **+0.185, p=0.004** |
| clc_all (warm-start) | all CLC, init=retest | 0.827 | **0.692** | **0.437** | +0.179, p=0.004 |
| clc_adult | adult CLC only | 0.799¹ | 0.536¹ | 0.272¹ | (adult subset) |
| clc_neonate | neonate CLC only | 0.839² | 0.725² | 0.443² | (neonate subset) |

¹ on the 7 adult held-out images. ² on the 8 neonate held-out images.

### 4a. clc_all from cpsam (Result 4, job 2646810)
Pooled 15 imgs: **0.834 / 0.657 / 0.410**. Adult 0.805, neonate 0.859 (AP@0.5).
Paired vs cpsam: Δ+0.185 AP@0.5, **12/15 wins, p=0.0043**. Vs retest: Δ+0.423,
14/15, p=0.0001. **CLC fine-tuning clearly works.**

### 4b. Warm-start does not beat from-cpsam (Result 5, job 2647218)
Init from `label_xfer_aug_retest`, refine regime (lr 1e-6, wd 0.05, 40 ep).
Pooled: 0.827 / 0.692 / 0.437. Direct paired vs from-cpsam: AP@0.5 Δ−0.006
(p=0.75, **tied**); AP@0.75 Δ+0.035 (p=0.055, marginal); AP@0.9 Δ+0.026 (p=0.68).
→ **Statistically equivalent; from-cpsam preferred (simpler, no Cunningham dep).**

### 4c. Age specialization does not help (Result 6, jobs 2647416/2647417)
Per-age models train on only 3–4 same-age animals. Direct paired vs `clc_all` on
the **same** images:

| subset | metric | age-specific | clc_all | Δ | p |
|---|---|---|---|---|---|
| adult (n=7) | AP@0.5 | 0.799 | 0.804 | −0.006 | 0.47 (tied) |
| adult | AP@0.75 | 0.536 | 0.584 | −0.048 | 0.078 |
| adult | AP@0.9 | 0.272 | 0.348 | −0.076 | 0.078 |
| neonate (n=8) | AP@0.5 | 0.839 | 0.859 | −0.020 | **0.016 (7/8 wins clc_all)** |
| neonate | AP@0.75 | 0.725 | 0.721 | +0.004 | 0.94 (tied) |
| neonate | AP@0.9 | 0.443 | 0.465 | −0.022 | 0.94 (tied) |

→ The single mixed-age `clc_all` model **matches or beats** age-specific models
on both ages. On this tiny dataset, **more data + cross-age regularization beats
same-age purity**; 3–4 animals starve the specialist (fold4's high test loss was
exactly this overfit).

---

## 4½. Annotation style: head vs full cell (the target question)

**The annotation target is the hair-cell HEAD, not the full cell.** Confirmed
2026-06-03:
- The **downstream IHC/OHC classifier performs better on the head** portion.
- **SAM segments the head more cleanly** (a compact, intensity-distinct region)
  than the full oblong cell.
- The **Cunningham label-transfer masks are already head-only** — a more
  spherical, slightly darker labeled region for both IHC (round) and OHC
  (oblong, head at one end).
- The **in-house CLC GT encloses the full oblong cell** — i.e. a *different and
  non-preferred* annotation standard.

### Consequences for the results above
1. **Every AP number in §3–§4 is scored against the full-cell GT** — the wrong
   reference for the head target. They rank models by "reproduce full-cell CLC
   GT," which is **not** the deployment objective.
2. **§3's "Cunningham model is bad on CLC" is mostly an artifact** (head masks
   vs full-cell GT → low IoU). Re-scored against head GT, it should improve a
   lot, and Cunningham+CLC combined training may become viable (styles would
   match).
3. **§4's `clc_all` (0.834) learned to reproduce full cells** — it is optimized
   for the wrong target. Its high AP means "matches full-cell GT well," not
   "produces good head masks."
4. The **CV machinery, the LOAO design, and the augmentation findings (§2)** are
   all target-agnostic and **still valid** — only the GT they score against
   needs to change.

### Path forward — convert CLC GT to head-only without manual redraw
Goal: turn the existing full-cell CLC masks into head-only masks
(auto-assisted), then re-run the experiments against the new GT.

Candidate methods (feasibility hinges on the head being a **visually distinct**
region — the head is only "slightly darker," so this must be verified first):
- **SAM point-prompting** — prompt Meta's `segment_anything` predictor (bundled
  with cellpose) at each cell's intensity-distinct head location with
  `multimask_output=True`; select the "subpart" candidate contained in the
  original cell mask. Carries over the existing mask's IHC/OHC class by overlap
  (same idea as the original `label_xfer`, but producing a tighter mask).
- **Reuse the Cunningham model** — it already emits head-style masks; match its
  outputs to the existing CLC masks by overlap. No prompt tuning; subject to the
  imaging domain gap.
- **Intensity thresholding within each mask** — keep the distinct-intensity head
  sub-region. Simplest; brittle if contrast is weak.

**Validation:** needs a small set of hand-drawn head masks (~5–10 cells) to
verify the auto-conversion — far less than redrawing all ~15×142 cells.

### Outcome (2026-06-03, `probes/head_pilot.py`) — BLOCKED, retarget abandoned
The feasibility pilot ran on the local CLC data and **refuted the premise**:

- **No visually distinct head.** CLC images are single-channel **uint8**, cells
  near-saturated (mean 208/255, ~8% clipped); within-cell intensity bimodality
  only ~0.60. There is no darker spherical head for SAM or a threshold to grab,
  even on clearly oblong cells.
- **The Cunningham model is not a clean head extractor either** — it predicts
  ~0.85× GT area (IQR 0.55–0.95) and *fragments* on oblong cells.
- **Probable cause:** uint8 quantization + saturation destroyed the subtle
  "slightly darker head" contrast (Cunningham source is uint16 and shows heads
  clearly).

**Consequence:** the head retarget is **blocked on 16-bit original acquisitions**
existing for CLC (open question for the lab). Until then the **full-cell target
is the target**, §3–§4's ranking stands as written, and everything downstream
(the deployed model, the IHC/OHC classifier, the GUI post-processing) is built on
full-cell masks. Figures: `head_pilot_adult.png`, `compare_cun_clc.png`,
`chan_probe_adult.png`.

---

## 5. Full-set era (67 images, 20 animals) — the final results

The 15-image set was ~22% of the labeled data. Rebuilt everything on all 67
images (`CLC_full`, grouped 5-fold, age-stratified). Narrative + per-experiment
detail is in [`progress.md`](progress.md); this is the scoreboard.

| arm | AP@0.5 | FN-recovery | FP-suppression | TP-retention | verdict |
|---|---|---|---|---|---|
| stock cpsam (the proposer) | 0.648¹ | 25.7% | 0.0% | 99.3% | the bar |
| **`clc_all_full` (control)** | **0.820** | **71.9%** | **89.6%** | **95.2%** | **the recipe that ships** |
| `clc_all_fnaug` (no cutout/blur) | flat | 72.1% | flat | flat | aug content is not the lever |
| `clc_adult_full` (adult-only) | 0.828² | 79.2%² | — | — | tied with mixed (p=0.68) |
| `clc_neonate_full` (neonate-only) | 0.804³ | 62.9%³ | — | — | **loses** to mixed 0.819 (p=0.046) |
| `clc_all_bndry` (boundary loss α=5) | 0.814 | 71.1% | 89.6% | 95.0% | no effect — not a loss problem |
| cpdino backbone | 0.819 | 71.2% | 89.7% | 95.1% | tied (p=0.85) |
| cpdino @384 crop | 0.818 | — | — | — | tied (p=0.72) |
| cpsam_v2 backbone | 0.809 | 69.0% | 89.4% | 95.0% | tied (p=0.117) |
| cpsam_v2 + fork online aug | 0.824 | — | — | — | tied (p=0.348) |

¹ measured on the early 15-image set; the full-set cpsam error-eval gives
FN-recovery 25.7% over 2990 FN / 5106 FP / 6774 TP. ² adult held-out only (n=32).
³ neonate held-out only (n=35). All Δ's are **paired per-image** vs the control.

### 5a. …and the data lever has now saturated (2026-07-31, +20 images)

The lab added 20 curated images (5 new animals + 3 re-imaged 4L; `CLC_full2`,
87 total). Tested **paired** — same recipe, same held-out images, fold assignment
pinned by `clc_split.py --extend`, control = the existing `clc_all_full_20260604`
fold models re-scored on the extended folds:

A third arm re-ran it with the batch's single out-of-domain **20x** acquisition
dropped from training, in case it was contaminating the set. All three arms
scored on the same 86 non-20x images:

| metric | A: 67 imgs (deployed) | B: 87 imgs | C: 86 imgs (no 20x) |
|---|---|---|---|
| AP@0.5 | 0.8165 | 0.8222 | 0.8194 |
| AP@0.75 | 0.5736 | 0.5771 | 0.5790 |
| AP@0.9 | 0.3296 | 0.3381 | 0.3383 |
| AP@0.5 — old 67 | 0.8203 | 0.8206 | 0.8193 |
| AP@0.5 — new 19 | 0.8032 | 0.8280 | 0.8200 |
| FN-recovery | 69.1% | 69.2% | 69.3% |
| FP-suppression | 89.4% | 89.3% | 89.5% |
| TP-retention | 95.4% | 95.8% | 95.6% |

Paired AP@0.5: **B vs A +0.0057 (p=0.94) · C vs A +0.0029 (p=0.64) · C vs B
−0.0028 (p=0.89)**.

**Tied on everything → the deployed model was not replaced.** Note the contrast
with the 15→67 rebuild below: that was *also* flat on AP but +16 pts on
FN-recovery, which is why both metrics get checked. Here neither moves, in either
variant. The returns are the expected shape — 4.5× more data bought 16 pts,
1.3× more buys nothing. Full record: `model_train_log.md` §4a (jobs 3346056 /
3346061 / 3346128 / 3346149 / 3346206).

Three details worth carrying forward:
- The new images cause **no regression** on the old 67 — they belong in the
  dataset for any future refresh.
- **The 20x acquisition was a bystander, not a contaminant.** It is the worst
  test image for the retrained model (0.592 → 0.427) because it is off-domain,
  but removing it from *training* changed nothing (C vs B p=0.89). A stray
  off-domain image among ~86 is harmless to keep.
- **20x is a separate training target.** cpsam has no diameter-rescale step, so
  one model can't straddle both magnifications for free; two unlabeled 20x tifs
  already sit in `cellpose_cc/adult(2)` if that becomes a goal.

### 5b. Re-tested again with the 2026-08 batch (98 images) — still nothing

4 more adult animals (a "myo 7e10 tomt overexp" prep) + 6 re-curated labels →
`CLC_full3`, 98 images / 29 animals, 20x excluded. Control re-scored from scratch
(6 images' GT had changed, so the previous CSVs were stale):

| metric | arm D (98 imgs) | control (67 imgs) | Δ | p |
|---|---|---|---|---|
| AP@0.5, all 98 | 0.7996 | 0.7907 | +0.0089 | **0.96** |
| AP@0.5, old 67 | 0.8169 | 0.8210 | −0.0041 | 0.33 |
| AP@0.5, 92 non-sparse | 0.8175 | 0.8169 | +0.0006 | 0.79 |
| FN-recovery | 69.7% | 69.3% | +0.4 pt | — |
| FP-suppression | 89.0% | 89.2% | −0.2 pt | — |

**Tied → prod unchanged (second confirmation).** Full record: `model_train_log.md`
§4b (jobs 3479313 / 3479318 / 3479399).

**But this round found a genuine gap.** Scored on the 31 images it has never
seen, `clc_all_full_deploy` gets **0.828** mean AP@0.5 on normal-density images —
exactly its certified level, including on the new prep and the new `GtGFP555`
stain — but only **0.177** on four severely degenerated cochleae (4–13 hair cells
left), where it predicts *zero* masks or over-detects on damaged tissue. Training
on two degeneration animals did **not** fix it (`8477 16khz` is still 0 detections
of 4). That is a distinct problem from the FN/merging limit above and needs its
own data or a prep-specific threshold — not another general retrain.

Two methodology lessons, both recorded in `model_train_log.md` §4b:
- **Diff `masks` / `ismanual` / `manual_changes`, never mtimes.** The GUI rewrites
  `_seg.npy` (colors, model_path, normalize_params, outlines) whenever the
  classifier runs; 31 of 32 files looked "updated" but only 6 had label changes.
  `ismanual` matters even when `masks` don't — it defines the FN/TP split.
- **Report GT-cell counts beside every subset.** Per-image AP means weight a
  4-cell image the same as a 238-cell one; the eye-catching "+0.136 on the
  degeneration subset" was one image gaining two detections, with 3 of 6 worse.

**Findings:**
1. **More data was the only lever that moved FN** — the same recipe on 67
   instead of 15 images took FN-recovery 55.9% → **71.9%** (+16 pts), with
   FP-suppression held (~90%) and TP-retention up (95.2%). **But it has since
   saturated**: 67 → 87 images changed nothing (§5a).
2. **Nothing else moved anything:** augmentation content, age specialization,
   cellprob/flow thresholds, boundary-weighted loss, and four backbone/crop-size
   variants are all statistically tied with the control. The residual ~28% FN is
   **merging of densely-packed touching cells** — a resolution/architecture
   limit, not a training knob.
3. **FP is solved** (~90% of cpsam's false masks suppressed) → no FP-rejection
   second model was needed for in-domain data. (The GUI's off-band reject panel
   is a *deployment* safety net for new/OOD images, not an in-domain fix.)
4. **Ship one mixed model**: `clc_all_full_deploy`, same recipe on all 67 images
   (`model_train_log.md` §1).

---

## 6. Conclusions

> **These conclusions are conditional on the full-cell GT** — which, after the
> §4½ head pilot came back blocked, **is** the target. They stood up on the full
> 67-image set (§5) and are the basis for the deployed model. Revisit only if
> 16-bit originals turn up and the head retarget becomes possible.

1. **Online augmentation replaces the offline D4 files** — parity at matched
   `nimg_per_epoch`, zero disk (§2). Always set `nimg_per_epoch` explicitly.
2. **The grouped-by-animal CV + pooled per-image AP machinery** is the harness
   for anything that follows; never split per image within an animal.
3. **A single `clc_all` model from cpsam wins** — AP@0.5 0.648 → 0.834 (15 img,
   p=0.004), holding at 0.820 on the full 67-image set. Warm-start from the
   Cunningham model is tied (not worth the dependency); per-age models don't help
   even when no longer data-starved (§5).
4. The "Cunningham model is bad on CLC" reading (§3) is largely an
   annotation-style artifact (head-style masks scored against full-cell GT) —
   but since the head retarget is blocked, it stays the practical conclusion:
   **don't deploy the Cunningham model on CLC data.**
5. **Ship `clc_all_full_deploy`** — the same recipe trained on all 67 images
   (`model_train_log.md` §1, `RUNBOOK.md` §4a).

### Open questions / caveats
- **AP@0.9 boundary tradeoff (full-cell GT):** every CLC model (~0.41–0.44 on
  the 15-img set, ~0.31 full-set) is *below* stock cpsam (0.510) at the strictest
  IoU. CLC fine-tuning detects far more cells (big AP@0.5/0.75 gains) but matches
  the loose hand-drawn full-cell boundaries less tightly. Given ~89% of GT
  boundaries *are* cpsam's own output, part of this gap is the GT provenance
  rather than a real regression — and AP@0.5 (correct detection) is what the
  downstream counting/classification actually needs.
- **Animal `5168` is the consistent outlier** (AP@0.5 0.28–0.49 across every
  recipe) — a single adult image dragging the adult mean. Worth a visual GT
  check (hard sample vs labeling issue) before trusting the adult numbers.
- **Residual ~28% FN is a hard limit**, not an untuned knob — six independent
  levers all came back flat (§5). Real headroom needs higher-bit-depth source
  images or super-resolution, not another config.
- The 15-image numbers (§1–§4) carry wide CIs; prefer the §5 full-set numbers.

---

## 7. Reproduce

> Full, current command set: [`RUNBOOK.md`](RUNBOOK.md). Per-model provenance
> (which config / job / data made which artifact): [`model_train_log.md`](model_train_log.md).

```bash
# split manifest (group by animal)
python clc_split.py --scheme loao --out_root ...   # -> clc_folds_loao.json
# materialize a recipe's per-fold dirs
python clc_cv.py materialize --manifest clc_folds_loao.json --recipe clc_all --out_root runs/clc_cv
# train+eval all folds (SLURM array); warm-start adds the init model path:
./submit_cv.sh clc_all                                   # from cpsam
CELLPOSE_TRAIN_CONFIG=$HOME/cellpose/trainer_warmstart.yaml ./submit_cv.sh clc_all <init_model_path>
# baseline (no training): score existing models on CLC originals
MODELS='cpsam,label_xfer_aug_retest' TEST_DIR='.../CLC/adult,.../CLC/neonate' sbatch --export=ALL,MODELS,TEST_DIR run_eval.slurm
# pool per-image AP + per-animal/age + paired test vs a baseline csv
python clc_cv.py aggregate --manifest clc_folds_loao.json --results_dir run_logs/<RUN_TAG> --baseline_csv run_logs/eval_<job>.csv
```

Scripts: `clc_split.py`, `clc_cv.py`, `run_clc_cv.slurm`, `submit_cv.sh`,
`eval_seg.py`, `trainer_slurm.py`, `trainer.yaml` / `trainer_warmstart.yaml`,
`aug_online.py`. Per-image CSVs: `~/cellpose/run_logs/<RUN_TAG>/eval_fold*.csv`.
