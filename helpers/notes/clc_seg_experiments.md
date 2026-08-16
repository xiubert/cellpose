# CLC mask-segmentation experiment plan

> **⚠️ HISTORICAL PLAN — executed and closed (2026-06-23).** This is the
> *design* doc, written when `label_xfer_aug_retest` was the incumbent and only
> 15 CLC images existed. It is kept for the methodology (§2 augmentation
> analysis, §3 infrastructure, §4 axis definitions), **not for its status**.
> Outcome: [`clc_seg_results.md`](clc_seg_results.md) §5 · status
> [`progress.md`](progress.md) · commands [`RUNBOOK.md`](RUNBOOK.md) · per-model
> provenance [`model_train_log.md`](model_train_log.md). **The deployed model is
> `clc_all_full_deploy`** (axis A4/A2-equivalent: all CLC, both ages, from
> stock cpsam), trained on 67 images. Axes A, B and C are all resolved; a
> backbone axis was added later and also came back flat.

**Goal:** improve Cellpose-SAM **cell-mask segmentation** on the in-house **CLC**
cochlear dataset (neonate + adult). Current best is `label_xfer_aug_retest`
(trained on Cunningham / PMID-38653806 transferred labels). This doc is the
plan + infrastructure for systematically beating it on CLC-domain data.

> Scope note: this is about **mask quality** (instance segmentation), not the
> IHC/OHC cell-type classifier (that pipeline lives in `helpers/ihc_ohc/`).

---

## 1. What we're working with (verified facts)

### Data — `/ix1/pcody/cellpose/data/CLC` (cluster)
- **15 unique cochleae total: 7 adult + 8 neonate.** Tiny.
- Every file is already **offline D4-augmented** (`<stem>_SV_<tag>`, ~8× each):
  49 adult + 64 neonate seg files. No plain originals retained for adult.
- Images: **1024×1024×3 uint16**, ~130 cells each. `class_map` present.
- **Not yet split** into train/test.
- Cunningham (`label_xfer`) data IS already split; PMID-38653806 transferred
  labels are the basis of the current best model.

### Consequences for methodology
- **15 images → a single held-out split is far too high-variance to rank
  models.** Use **GroupKFold-by-cochlea** (group key = filename before `_SV`),
  stratified so each fold carries both adult and neonate. Mirror the
  leak-free, group-by-source discipline the IHC/OHC side already uses.
- **Evaluate held-out folds on un-augmented originals only.** Augmented copies
  in a test fold inflate the metric. (Adult currently has no plain originals on
  disk — regenerate them or hold out one full `_SV_*` set per cochlea and dedupe.)
- Prior CLC runs already exist in `models/tests/` but are **unscored**:
  `CLC_aug_adult_20250504`, `CLC_small_set`, `label_xfer_aug_plusCLC`,
  `label_xfer_aug_wAdultCLCx2`, `label_xfer_plusCLC`, `label_xfer_plusCLC_2`.
  Backfilling their scores may already answer some questions for free.

### Metric
`cellpose.metrics.average_precision(masks_true, masks_pred, threshold=[0.5,0.75,0.9])`
— report **AP@0.5 (primary), AP@0.75, AP@0.9** plus mean true/pred cell counts
(count drift is the fastest tell of over/under-segmentation).

---

## 2. Augmentation — what's redundant, what's additive

cellpose's trainer already augments **geometrically** every epoch
(`random_rotate_and_resize`, transforms.py): continuous rotation (0–2π), random
flip, random scale (`scale_range`), random 256² crop.

| Offline `augment.py` does | Built-in trainer does | Verdict |
|---|---|---|
| D4 rot90/180/270, fliph/flipv | continuous rotate + flip every epoch | **redundant** |
| (scale) | random scale (`scale_range`) | covered |
| `--intensity` brightness/gamma | nothing | additive → online |
| (none) cutout / square dropout | nothing | additive → online |
| (none) noise / blur | nothing | additive → online |

So the offline `_SV_*` D4 copies add no geometric information — only sampling
multiplicity and disk bloat. The genuinely-new augmentation is **photometric +
occlusion**, and it's now done **online** (zero disk), see §3.

---

## 3. Infrastructure already built

- **`helpers/aug_online.py`** — `make(cfg)` returns an image-only
  `img_transform` callable: `intensity` (scale+gamma), `cutout` (square
  dropout), `gauss_noise`, `gauss_blur`. Per-image, **labels never touched**,
  nothing written to disk. Returns `None` when nothing is enabled.
- **`cellpose/train.py`** — minimal generic hook: `train_seg(..., img_transform
  =None)` applied to the augmented batch `imgi` in the **train** loop only
  (not test/val).
- **`trainer.yaml`** — single source of truth for a run. Three sections:
  ```yaml
  data:                # dataset for the run (frozen into each run's config)
    source:    /ix1/pcody/cellpose/data/PMID_38653806/Cunningham  # has train/ + test/
    model_dst: /ix1/pcody/cellpose/models                          # where models copy back to

  train:
    weight_decay:   0.1
    learning_rate:  1.0e-5
    n_epochs:       95
    batch_size:     8
    nimg_per_epoch: 750  # crops sampled per epoch — SEE §5/§Results; cellpose
                         # defaults this to the #train files, so originals-only
                         # runs get far fewer steps unless set explicitly.
    model_name:     <unique per run>

  augment:             # online image-only aug; omit a section to disable it
    seed: 0
    intensity:   {p: 0.5, scale: [0.75, 1.25], gamma: [0.8, 1.25]}
    cutout:      {p: 0.5, n: [1, 3], size_frac: [0.05, 0.15], fill: mean}
    gauss_noise: {p: 0.3, sigma_frac: 0.05}
    gauss_blur:  {p: 0.2, sigma: [0.5, 1.5]}
  ```
- **`trainer_slurm.py`** — reads `train:` + `augment:`, passes `img_transform`
  and `nimg_per_epoch` to `train_seg`. `run_trainer.slurm` reads `data.source`
  (bash-side, via PyYAML) to stage scratch; `CELLPOSE_DATA_SRC` overrides.
- **`helpers/eval_seg.py` + `helpers/run_eval.slurm`** — head-to-head eval:
  runs N models on a test dir, reports **AP@[0.5/0.75/0.9] + mean true/pred
  cell counts** (`cellpose.metrics.average_precision`), writes a per-image CSV.
  Skips augmented stems (originals-only). Submit e.g.
  `MODELS='a,b,c' sbatch --export=ALL,MODELS run_eval.slurm`.

### Deployment (important)
Training imports the **pip-installed** cellpose, **not** the vendored
`cellpose_git/cellpose/` source. After any edit under `cellpose/` or to the
trainer scripts, push it into the runtimes:
- `./update_container.sh` — overwrites the container's installed package.
- `./update_cluster.sh [host]` — cluster analogue (resolves the conda-env
  cellpose dir; copies trainer scripts + `aug_online.py` to `~/cellpose/`).
  `DRY_RUN=1` to preview.

Run everything inside the container / via SLURM in a **screen session** so
progress is monitorable.

---

## 3a. Results so far (Cunningham held-out test, 23 originals)

First experiments validated the online-augmentation infrastructure on the
**Cunningham label-transfer set itself** (not yet CLC), against the current
best `label_xfer_aug_retest` (trained on the old offline-D4 files).

| model | train data / aug | AP@0.5 | AP@0.75 | AP@0.9 | n_pred (true 62.0) |
|---|---|---|---|---|---|
| `label_xfer_aug_retest` | 93×8 **offline D4** files, 95 ep | **0.808** | **0.601** | 0.305 | 58.7 |
| `label_xfer_online_aug` | 93 originals, **online aug**, 95 ep, `nimg_per_epoch`=default(93) | 0.735 | 0.570 | 0.238 | 53.2 |
| `label_xfer_online_aug_nimg750` | 93 originals, **online aug**, 95 ep, **`nimg_per_epoch`=750** | 0.787 | 0.595 | **0.304** | 57.8 |

**Takeaways**
1. **`nimg_per_epoch` was a hidden confound.** cellpose defaults it to the
   #train files. With offline D4 deleted (93 files vs the old ~744), the first
   online run saw **~8× fewer crops/epoch** → undertrained, under-segmenting
   (53.2 vs 62 true cells), AP@0.5 0.735.
2. **Budget-matched (`nimg_per_epoch`=750 ≈ 93×8), online aug reaches ~parity:**
   AP@0.5 0.787, **tied at AP@0.9** (0.304 vs 0.305), under-segmentation fixed
   (57.8 ≈ retest 58.7). Test loss also dropped 0.364→0.261.
3. **Online aug is a sound zero-disk replacement for the offline D4 files.**
   The geometric D4 copies were redundant (cellpose augments geometry on the
   fly); the photometric/occlusion ops now run online. Residual ~0.02 gap at
   AP@0.5/0.75 is within noise on 23 images (could probe with softer aug).

**Operational rule:** when training on originals-only, **always set
`nimg_per_epoch` explicitly** (≈ desired effective dataset size) — critical for
CLC (only 15 cochleae) where the default would be tiny.

Run records: train jobs 2644575 (default) / 2644707 (nimg750); eval jobs
2644683, 2646554; per-image CSVs in `~/cellpose/run_logs/eval_<jobid>.csv`.

---

## 4. Experiment matrix

Three axes. Rank by **CLC-test AP@0.5 under GroupKFold-by-cochlea CV**. Pick
hyperparameters by CV/sweep only; treat any final held-out read as a single
locked number (same test discipline as the IHC/OHC pipeline).

### Axis A — data composition (the core question)
| ID | Train set | Init | Question |
|---|---|---|---|
| A0 | none | stock Cellpose-SAM | floor |
| A1 | none | `label_xfer_aug_retest` | does Cunningham transfer already suffice? |
| A2 | CLC adult only | SAM | adult domain alone |
| A3 | CLC neonate only | SAM | neonate alone (do ages need separate models?) |
| A4 | CLC adult + neonate | SAM | is one CLC model enough? |
| A5 | Cunningham + CLC pooled | SAM | does keeping Cunningham help or dilute? |
| **A6** | **CLC (all)** | **warm-start `label_xfer_aug_retest`** | **sequential domain adaptation vs pooling — most likely winner on 15 imgs** |
| A7 | adult-only / neonate-only | warm-start | only if A4 shows age conflict |

### Axis B — augmentation (run on the best 1–2 data configs from A)
| ID | Augmentation | Tests |
|---|---|---|
| B0 | built-in geometric only (no offline D4, no `augment:`) | is offline D4 redundant? (expected: yes) |
| B1 | + intensity/gamma | confocal exposure/bleaching variation |
| B2 | + **cutout** | occlusion / saturation / debris robustness |
| B3 | + noise/blur | detector noise + PSF |
| B4 | B1+B2+B3 | full regime |

Cutout caveat: a dropout square that erases a cell interior while its label
stays positive teaches "predict a cell from background" — that's the intended
robustness signal, but keep squares modest (≤ ~1 cell diameter, low count) or
it degrades. `size_frac: [0.05, 0.15]` of 1024 px ≈ 51–154 px is a sane start.

### Axis C — hyperparameters (two regimes, not a full grid)
**From-SAM / pooled training (A2–A5):** current defaults are reasonable —
```yaml
learning_rate: 1.0e-5
weight_decay:  0.1
n_epochs:      95
batch_size:    8
# consider nimg_per_epoch: 200–400 so an "epoch" isn't ~12 steps
```
**Warm-start refine on CLC (A6 — the important one):**
```yaml
learning_rate: 1.0e-6   # 10× lower; avoids erasing Cunningham knowledge
weight_decay:  0.05     # 0.1 if overfit appears (CLC AP up but count drift)
n_epochs:      30       # watch test-fold AP for early plateau
batch_size:    8
nimg_per_epoch: 200     # decouple epoch from the ~12-image train fold
```
**One deliberate `scale_range` sweep** (0.3 / 0.5 / 0.7): if CLC cell pixel-size
differs from Cunningham, this is the single geometric knob most likely to move
AP. Lower = trust CLC scale; higher = more scale jitter.

---

## 5. Execution order
0. **DONE** — eval harness built (`eval_seg.py`) + online-aug validated on
   Cunningham (§3a). Online aug = parity with offline D4 at matched
   `nimg_per_epoch`.
1. **Build CV split** (GroupKFold-by-cochlea) + score baselines **A0, A1**, and
   **backfill scores for the existing `models/tests/` runs** with `eval_seg.py`
   — may already answer parts of Axis A before any new GPU time.
2. **Axis A** at default hyperparams → pick winning data composition.
   (Set `nimg_per_epoch` explicitly — see §3a operational rule.)
3. **Axis C** regime tuning on that winner (A6 LR/epochs + `scale_range` sweep).
4. **Axis B** augmentation on the tuned winner — where cutout earns or doesn't
   earn its place.

Each run → fresh `model_name` + frozen `trainer.yaml` config in
`run_logs/cellpose_train_<jobid>.config.yaml` (SLURM already does this). One
submission should loop the 5 CV folds.

---

## 6. TODO — all closed
- [x] `eval_seg.py` — AP@[0.5/0.75/0.9] + mean cell counts → CSV (+ `run_eval.slurm`).
- [x] `nimg_per_epoch` exposed in trainer.yaml/trainer_slurm.py; `data:` block + bash staging.
- [x] Online aug validated vs offline D4 on Cunningham (§3a).
- [x] GroupKFold-**by-animal** CLC split (`clc_split.py`; animal, not cochlea-image
      — tonotopic regions share a cochlea). Originals-only test fold.
- [x] Baseline A0 (stock cpsam 0.648) + A1 (`label_xfer_aug_retest` 0.411) on CLC — job 2646778.
- [~] Backfill scores for the old `models/tests/` runs — **skipped**: superseded
      by the from-scratch CV arms, which cover the same questions leak-free.
- [x] CV-fold loop — done as a SLURM **array** (`run_clc_cv.slurm` + `submit_cv.sh`),
      not a loop inside `run_trainer.slurm`.
- [x] Residual-gap probe → became the FN-aug arm (`trainer_fnaug.yaml`, drop
      cutout+blur): no effect.

Added after this plan was written (all flat): boundary-weighted loss
(`trainer_bndry.yaml`), cellprob/flow threshold sweep, and a **backbone axis**
(cpdino, cpdino@384, cpsam_v2). See `clc_seg_results.md` §5.

---

## 7. References
- Augmentation hook: `helpers/aug_online.py`, `cellpose/train.py`
  (`img_transform`), `helpers/trainer.yaml` (`augment:` block).
- Config / training: `helpers/trainer.yaml` (`data:` + `train:` incl.
  `nimg_per_epoch` + `augment:`), `helpers/trainer_slurm.py`.
- Eval: `helpers/eval_seg.py`, `helpers/run_eval.slurm` (AP@IoU + cell counts).
- Deploy: `update_container.sh`, `update_cluster.sh`.
- SLURM workflow: `helpers/slurm.md`, `helpers/run_trainer.slurm`,
  `helpers/submit.sh`.
- Label transfer / current best model provenance: `helpers/legacy/label_xfer.md`,
  top-level `CLAUDE.md`.
