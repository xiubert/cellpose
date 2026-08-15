# IHC vs OHC Classifier

A tiny CNN that labels each Cellpose-segmented hair cell as an **inner**
(IHC) or **outer** (OHC) hair cell, from the MYO7A channel plus the
instance's own mask. Two scripts + one config:

| File | Phase | Role |
|---|---|---|
| `ihc_ohc_crops.py` | 1 — data prep / inference cropping | Expand each instance bbox, pad, add the target-mask channel, resize → fixed crops |
| `ihc_ohc_classifier.py` | 2 — model | `TinyHCNet` `sweep` / `cv` / `train` / `predict`; reuses Phase-1 cropping at inference |
| `configs/cnn.yaml` | — | all classifier hyperparameters (sweep / CV / train) |

All of the above live in [`helpers/ihc_ohc/`](.) and run inside the
cellpose container as `python3 /helpers/ihc_ohc/<script>.py …`. Run
artifacts (best.pt, history.json, sweep_results.json, the crops/geom
`.npz` caches, …) go to `runs/` next to the code — see
*[Run-artifact layout](#run-artifact-layout)* below.

## Architecture at a glance

The deployed classifier is a **late fusion** of two decorrelated models —
one reads MYO7A appearance, the other reads where each cell sits in the
organ of Corti. Their errors are largely independent, so the calibrated,
threshold-tuned weighted average lifts every metric over either alone
with no trade-off (CNN 0.951 → geom 0.961 → fused **0.977** bal_acc on
Cunningham held-out test).

```
Cellpose-SAM mask  ─┐
                    ├──► (1) TinyHCNet CNN                         ─┐
MYO7A TIF          ─┤        on (MYO7A crop, target-mask)           │
                    │        → P(IHC) ───► class_prob               │
                    │                                                │
                    └──► (2) Geometric                              ─┤
                             PCA centerline + regionprops +         │  weighted
                             kNN graph features → logreg            │   mean
                             → P(IHC) ───► class_prob_geom          │  (w = 0.40)
                                                                    │
                  ┌──► (3) Late fusion (frozen on OOF train)        ◄┘
                  │        w·P_cnn + (1-w)·P_geom ≥ 0.440
                  │        ───► class_map_fused
                  ▼
              <stem>_pred.npy  (sidecar; seg untouched)
                    │
                    └──► GUI: user clicks/region-selects to correct
                              ───► class_map_user (sparse, persistent;
                                   wins per cell on next train rebuild)
```

- **(1) TinyHCNet** — 3 conv blocks + GroupNorm + GAP → 2-class logits;
  ~72 k params, trains in minutes on an RTX 2060 SUPER. See *Model* below
  for why GroupNorm + balanced sampler + EMA-bal_acc selection.
- **(2) Geometric** — 21-D per-cell vector: regionprops shape, signed
  perpendicular distance to a per-image **PCA-regression** centerline
  (the dominant feature — IHC sit on the line, OHC ~3 cells off it),
  kNN-graph anisotropy, hull-edge distance. Logreg (`class_weight:
  balanced`). Torch-free.
- **(3) Fusion** — the CNN's `class_prob` and the geom's `proba_ihc`
  averaged with a single weight `w` picked **out-of-fold** (`fuse.cnn_oof:
  true`, so the CNN is retrained per fold rather than read in-sample).
  McNemar fused-vs-CNN on the test set: 55 cells fixed vs 23 lost,
  χ²=12.3, p ≈ 4e-4 — the gain is real, not noise.

## Labels in / labels out

**In — what the training pipeline reads.** Per cochlea, training expects
one `<stem>_seg.npy` + paired `<stem>.tif` in `train_dir/` (and the same
shape for `test_dir/`). Per-cell labels are resolved by
**`resolve_training_label_map(seg, seg_path, data_dir)`** in
[`ihc_ohc_crops.py`](ihc_ohc_crops.py) — used by *both* the crops builder
and the geom builder, so the two stay aligned. It merges three sources;
later rows override earlier per cell:

| Order | Source | Where | Note |
|---|---|---|---|
| 1 | seg dict | `seg["class_map"]` = `{int: "IHC"\|"OHC"}` | the from-scratch GT — written by `label_xfer.py` from VOC XML; `augment.py` propagates to D4 copies. |
| 2 | VOC XML | `<base>.xml` (PASCAL VOC bndbox per cell) | fallback when `seg["class_map"]` is missing — each mask gets the class of the box it overlaps most (degenerate <5 px boxes dropped). |
| 3 | sidecar | `<stem>_pred.npy → class_map_user` = `{int: "IHC"\|"OHC"}` | **the GUI's human-in-the-loop channel** — sparse (only the cells the user clicked); *wins per cell* over both above. |

`<base>` for the XML lookup drops the `_myo7a` suffix and any D4 aug
tag, so one XML maps to every variant of one cochlea. Cells with **no**
source after merging are silently skipped at crop/geom-build time and
counted in the per-image summary. The provenance tag (`"seg"`, `"xml"`,
`"user"`, `"seg+user"`, `"xml+user"`) is recorded so you can audit later
which cochleae contributed which kind of label.

**Out — what `predict` writes.** All predictions go to a
**`<stem>_pred.npy` sidecar** next to the seg, never into the dataset
seg itself. Atomic write (tmp file + `os.replace`). Eight keys total
across the three classifiers and the GUI label channel; each is
`{int_mask_id: …}`:

| Key | Producer | Type | Coverage | Meaning |
|---|---|---|---|---|
| `class_map_pred`   | CNN  | `str`   | every mask | `"IHC"` or `"OHC"` |
| `class_prob`       | CNN  | `float` | every mask | P of the predicted class (≥ 0.5 by definition) |
| `class_map_geom`   | geom | `str`   | every mask | `"IHC"` or `"OHC"` |
| `class_prob_geom`  | geom | `float` | every mask | P of the predicted class |
| `geom_flag`        | geom | `int`   | every mask | bitmask: ≥ 1 ⇒ off-axis / sparse / extrapolated / too-few-cells — human review hint |
| `class_map_fused`  | fuse | `str`   | every mask | **the deployed label** — what the GUI / `plot` colour-codes by default |
| `class_prob_fused` | fuse | `float` | every mask | P of the predicted class |
| `class_map_user`   | **GUI** | `str` | **sparse — only cells the user clicked** | hand-applied class; **survives `predict` re-runs** (GUI: `run_celltype` snapshots-then-wipes; CLI: `update_pred` merges so user labels are never overwritten) and feeds the next training rebuild via `resolve_training_label_map`. |

The GUI's celltype handler walks `class_map_user → class_map_fused →
class_map_geom → class_map_pred` and uses whichever is present — so user
clicks always override model output in the display. `load_pred(seg_path,
seg)` reads the sidecar (with a transparent in-seg fallback for
pre-sidecar data); `set_user_label` / `set_user_labels_bulk` in
[`cellpose/gui/celltype.py`](../../cellpose/gui/celltype.py) are how the
GUI persists clicks atomically. See also *Prediction sidecars* below.

## Pipeline

```
Image (MYO7A)  +  Cellpose mask  +  class_map (IHC/OHC, or VOC XML fallback)
        │
        ├─ per instance: bbox → expand (pad_frac·max(h,w)) → zero/mean pad
        ├─ + target-instance binary mask as channel 2
        ├─ resize → (2, 64, 64) float32
        ▼
   crops_*.npz  ──►  TinyHCNet  ──►  IHC / OHC probability
```

Input is the `label_xfer.py` / `augment.py` output: `<stem>_myo7a.tif` +
`<stem>_myo7a_seg.npy` with keys `masks` and `class_map`
(`{mask_id: "IHC"|"OHC"}`).

## Where labels come from

`class_map` is present in every Cunningham `_seg.npy` (originals **and** D4
copies) and is the primary source. If it is ever missing, the builder falls
back to the original VOC XML (`<NNN>_cunningham_mouse_confocal.xml`) and
assigns each mask the class of the bounding box it overlaps most —
reconstructing the same correspondence `label_xfer.py` built (degenerate
< 5 px boxes skipped, so counts line up: image 000 → 18 IHC / 52 OHC, the
one dropped IHC being a 0-height box).

## Augmented copies

`augment.py`'s D4 variants (`_rot90`, `_rot180`, `_fliph`, …) preserve
`class_map` exactly. The classifier augments on the fly (0–360° rotation,
flips, jitter), so the builder **skips on-disk copies by default** to avoid
inflating epochs and leaking cells across the train/val split. Pass
`--include-augmented` to ingest them; they are grouped by source-image id
either way so a cochlea never straddles the split.

## Crop design

- **bbox** from `masks == cell_id`, expanded by `pad_frac · max(h, w)` per
  side (default 0.5 → ~one cell-width of context). `--pad_px` for a fixed
  pixel pad instead.
- **edge handling**: `crop_pad` allocates the target window and copies only
  the in-bounds overlap; everything outside the image is the fill value
  (per-image **mean** by default — set `--pad_value 0` for zeros). Robust to
  windows partly or wholly outside the image.
- **mask channel**: the binary mask of *that one instance only*, so the net
  knows which cell in the padded crop to classify. Soft (bilinear) by
  default; `--hard_mask` to binarise.
- **resize**: `cv2` — `INTER_AREA` for the down-sampled image channel,
  `INTER_LINEAR` for the mask.
- intensities are kept **raw**; per-channel z-scoring lives in the
  classifier so training and inference share one normalisation.

## Model

`TinyHCNet`: 3 conv blocks (16→32→64, two 3×3 convs + **GroupNorm** + ReLU
each, 2× max-pool) → global average pool → dropout → linear(2). ~72k
params; trains in minutes on the RTX 2060 SUPER.

Imbalance & stability (tuned from the smoke test, see below):

- **Balanced batch sampler** (`WeightedRandomSampler`, ~50/50 batches) — the
  main IHC/OHC imbalance lever; also feeds GroupNorm a class-balanced view
  every step. Default `--sampler balanced`; pair with `--class_weight none`
  (don't double-correct).
- **GroupNorm, not BatchNorm** — small class-imbalanced batches make BN's
  train/eval running stats diverge, which caused the violent val-loss
  oscillation in the first smoke run.
- **AdamW + linear-warmup → cosine** LR (`--warmup_epochs`), replacing
  ReduceLROnPlateau. Set `--epochs` to a *realistic* horizon (~40, not 80):
  the cosine must fully anneal within the run or the tail stays noisy and
  early stopping cuts it off mid-oscillation.
- **Selection & early stop on EMA-smoothed balanced accuracy**
  (`--select_ema`, default 0.5), not val loss and not raw bal_acc — the
  ~19-image val set spikes ±0.03 epoch-to-epoch, so picking the single
  best epoch locks onto luck. The EMA tracks the true level.
- Optional **focal loss** (`--loss focal --focal_gamma`) as a second
  minority lever.
- On-the-fly augmentation: photometric jitter (image channel), 0–360°
  rotation, flips, ±3 px translate, random erasing.

### Effect on the Cunningham test set

| Config | acc | bal_acc | IHC rec | OHC rec | macro-F1 |
|---|---|---|---|---|---|
| BN + CE-weight + val-loss select | 0.80 | 0.74 | 0.62 | 0.86 | 0.74 |
| GroupNorm + balanced sampler + raw bal-acc, 12 ep | 0.90 | 0.92 | 0.96 | 0.88 | 0.88 |
| + 40-ep matched cosine, batch 256, EMA select | **0.92** | **0.94** | **0.97** | **0.91** | **0.90** |

The minority class was the problem (IHC recall 0.62 despite 3:1 loss
weighting); balanced batches + GroupNorm fixed it with no OHC trade-off.
Matching the cosine horizon to the run, a larger batch, and EMA selection
then stabilised the tail and lifted every metric further — no trade-off.

## Dependencies

Beyond stock `pip install cellpose` (numpy, scipy, tifffile, opencv,
torch, torchvision, segment_anything, …), the helpers add four extras —
listed and pinned in [`requirements.txt`](requirements.txt):

| Package | Why |
|---|---|
| **scikit-learn** | `GroupKFold` for the CNN CV splits + `LogisticRegression`/`GradientBoostingClassifier`/`GaussianMixture`/`StandardScaler`/`KDTree` for the geom classifier + the fusion stacker. |
| **scikit-image** | `regionprops_table` for the per-mask shape features — the geom pipeline's headline dependency. |
| **PyYAML** | every CLI takes `--config <yaml>` as single source of truth, and the GUI celltype manifest is yaml too. |
| **matplotlib** | crop/QC/mask previews, training-log plots, `ihc_ohc_pipeline.py plot`. Uses the `Agg` backend everywhere so no display server is needed. |

Install into the live container:

```bash
podman exec cellpose pip install -r /helpers/ihc_ohc/requirements.txt
```

Or, to bake into a rebuilt image, add to the Dockerfile:

```dockerfile
RUN pip install --no-cache-dir scikit-learn==1.7.2 scikit-image==0.25.2 \
                                PyYAML==6.0.3 matplotlib==3.10.9
```

**Container note:** the cellpose container's `/dev/shm` is only 63 MB, too
small for PyTorch DataLoader worker IPC — keep `train.workers: 0` (the
default). The crops are already an in-RAM numpy array, so 0 is also fast.
Raising it bus-errors unless the container is restarted with a larger
`--shm-size`.

## Configuration

All classifier hyperparameters live in a YAML config, **not** CLI flags —
runs are reproducible and sweeps are declarative. See
[`configs/cnn.yaml`](configs/cnn.yaml) for the annotated schema
(`data:` / `train:` / `cv:` / `sweep:`). A partial file is valid (missing
keys fall back to `DEFAULT_CONFIG`); unknown keys are rejected; the resolved
config is echoed at the start of every run AND frozen as `config.yaml`
inside the run's artifact dir (see *Run-artifact layout* below). Crop
building (`ihc_ohc_crops.py`) stays CLI-flag driven (it is run once).

## Run-artifact layout

Every `train` / `sweep` / `fuse` invocation writes to a fresh
**timestamped subdir** under `data.out_dir` (default
`/helpers/ihc_ohc/runs/`) — previous runs are preserved automatically:

```
helpers/ihc_ohc/
  ihc_ohc_*.py
  configs/{cnn,geom}.yaml
  ihc_ohc.md
  runs/                         (gitignored)
    cache/                      preprocessing caches (named, not stamped)
      crops_train.npz, crops_test.npz, geom_train.npz, geom_test.npz
    20260520-104530_train-cnn/  best.pt, history.json,
                                test_misclassified.png, config.yaml
    20260520-104530_train-geom/ geom_best.pkl, test_report.json, config.yaml
    20260520-104530_fuse/       fuse.pkl, config.yaml
    20260519-113047_sweep-cnn/  sweep_results.json, config.yaml
    …
```

`config.yaml` is the fully-resolved config at run time, so any past run is
reproducible from its own folder. `ihc_ohc_pipeline.py train` reuses one
timestamp across the three steps it runs so they group visually; the
underlying CLIs auto-stamp when called directly and accept `--run_dir
<path>` for explicit placement.

To use a previous run for inference, point the config's
`data.cnn_ckpt` / `data.geom_ckpt` / `data.fuse_ckpt` at the desired files
(or pass `--cnn_ckpt` / `--geom_ckpt` / `--fuse_ckpt` to
`ihc_ohc_pipeline.py predict`).

## Usage

Run inside the cellpose container (`podman exec cellpose …`). Mounts:
`/data` = `/media/DATA/Chris/cellpose2D`, `/helpers` = the host
`cellpose_git/helpers/` dir; this folder is `/helpers/ihc_ohc/`.

```bash
# 1. build crops (image-level split is implicit: train/ vs test/ dirs)
python3 /helpers/ihc_ohc/ihc_ohc_crops.py \
    --data_dir /data/to_zip/hcat-data/Confocal/Cunningham/traintest/train \
    --out /helpers/ihc_ohc/runs/cache/crops_train.npz \
    --preview /helpers/ihc_ohc/runs/cache/crops_train_preview.png
python3 /helpers/ihc_ohc/ihc_ohc_crops.py \
    --data_dir /data/to_zip/hcat-data/Confocal/Cunningham/traintest/test \
    --out /helpers/ihc_ohc/runs/cache/crops_test.npz

# 2. eyeball runs/cache/crops_train_preview.png BEFORE training

# 3a. (optional) hyperparameter sweep — edit `sweep:` in the config first
python3 /helpers/ihc_ohc/ihc_ohc_classifier.py sweep --config /helpers/ihc_ohc/configs/cnn.yaml

# 3b. (optional) cross-validate the chosen config (generalisation estimate)
python3 /helpers/ihc_ohc/ihc_ohc_classifier.py cv    --config /helpers/ihc_ohc/configs/cnn.yaml

# 4. train the final model (one split → checkpoint + held-out test report)
python3 /helpers/ihc_ohc/ihc_ohc_classifier.py train --config /helpers/ihc_ohc/configs/cnn.yaml

# 5. predict on a segmentation (optionally write predictions back in)
python3 /helpers/ihc_ohc/ihc_ohc_classifier.py predict \
    --ckpt /helpers/ihc_ohc/runs/<stamp>_train-cnn/best.pt \
    --seg  /data/.../000_cunningham_mouse_confocal_myo7a_seg.npy --write
```

Typical flow: **sweep → pick the top row → copy its values into `train:` →
cv → train**. Outputs in the new `runs/<stamp>_train-cnn/`: `best.pt`
(weights + norm stats + crop params + resolved `train_cfg`),
`history.json`, a test confusion matrix (stdout), `test_misclassified.png`,
and a frozen `config.yaml`. `sweep` writes `sweep_results.json` +
`config.yaml` into `runs/<stamp>_sweep-cnn/`.

## Knobs

Crop builder (`ihc_ohc_crops.py`, CLI flags):

| Flag | Default | Note |
|---|---|---|
| `--out_size` | 64 | crop side |
| `--pad_frac` / `--pad_px` | 0.5 / — | context padding |
| `--pad_value` | mean | `0` for zero-pad |
| `--hard_mask` | off | binarise mask channel |
| `--include-augmented` | off | also use on-disk D4 copies |

Classifier (`configs/cnn.yaml`):

| Key | Default | Note |
|---|---|---|
| `cv.folds` / `cv.val_frac` | 5 / 0.2 | GroupKFold by image; val_frac used if folds≤1 |
| `train.epochs` / `train.patience` | 40 / 15 | cosine horizon; early stop on smoothed bal_acc |
| `train.batch_size` | 256 | GroupNorm is batch-insensitive; smooths the curve |
| `train.select_ema` | 0.5 | EMA factor for the selection metric; 0 = raw |
| `train.sampler` | balanced | `balanced` (~50/50 batches) or `random` |
| `train.class_weight` | none | `balanced` re-enables CE/focal weighting |
| `train.loss` / `train.focal_gamma` | ce / 1.5 | `focal` for the focal-loss lever |
| `train.lr` / `train.warmup_epochs` | 0.001 / 4 | AdamW, linear warmup → cosine decay |
| `sweep.<train key>` | — | list of candidates → CV-scored grid |

## Inference integration

`predict_seg()` (and `predict` CLI) writes `class_map_pred` /
`class_prob` to a **prediction sidecar** (see *Prediction sidecars*
below), so the dataset `_seg.npy` is never mutated — `plot_boxes.py` and
the Cellpose GUI still work, and the probability is available as a
confidence score to fuse with the geometric classifier (weighted vote /
tiebreaker).

## Prediction sidecars

Predictions never live inside the dataset seg. Every classifier writes
to `<stem>_pred.npy` next to the seg, holding a dict of the prediction
keys (`class_map_pred`, `class_prob`, `class_map_geom`,
`class_prob_geom`, `geom_flag`, `class_map_fused`, `class_prob_fused`).
Writes are atomic-ish (tmp file + `os.replace`). The helpers live in
[`ihc_ohc_crops.py`](ihc_ohc_crops.py): `pred_path`, `load_pred`,
`update_pred`.

`load_pred(seg_path, seg)` prefers the sidecar and falls back to the
same keys *inside* the seg, so segs from earlier runs (when predictions
were stored in-line) still work — no forced migration. To clean up
legacy in-seg keys on demand:

```bash
python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py migrate-preds --dir <data_dir> --dry-run
python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py migrate-preds --dir <data_dir>
```

The migrator only strips keys from the seg *after* the sidecar write
succeeds — never silently. This is the only operation that mutates a
dataset seg, and it's opt-in.

---

# Geometric classifier + CNN fusion (Path A)

A second, **training-free-capable** classifier from pure mask geometry,
late-fused with the CNN. The CNN reads MYO7A *appearance*; this reads
*where each cell sits in the organ of Corti*. Their errors are largely
independent, so averaging them is the cheapest real accuracy gain
available — and it needs **no change to the tuned CNN** (it already writes
`class_prob` for exactly this).

| File | Phase | Role |
|---|---|---|
| `ihc_ohc_geom.py` | 1 — features | `geom_features_for_seg` (one source of truth, builder + inference) → 21-D per-cell vector; CLI builder → `geom_*.npz`; QC overlay |
| `ihc_ohc_geom_clf.py` | 2 — model + fusion | `rule` / `cv` / `train` / `sweep` / `fuse` / `predict` |
| `configs/geom.yaml` | — | all geom + fusion hyperparameters |

## The signal

The cochlea is long and thin: PC1 of the centroids runs along its length,
PC2 across the IHC-row + 3-OHC-row band. Fit the across-coordinate as a
**low-degree polynomial of the along-coordinate** (a regression centerline,
*not* an interpolating spline — a spline threads the 4-row band and sits
≈0 px from every cell, destroying the signal). The residual `w − g(t)` is
the across-band offset, and it alone separates IHC from OHC at ≈**0.998**
per image. The remaining 20 features (shape via `skimage.regionprops`,
kNN-graph spacing/anisotropy, hull-edge proximity) sharpen the boundary
and feed the learned model.

## Feature groups (21-D, names in `FEATURE_NAMES`)

- **shape** — area, perimeter, eccentricity, solidity, extent, axis
  lengths, axis ratio, equiv. diameter. Sizes ÷ per-image median cell
  diameter `D` → magnification-invariant.
- **axis** — signed across-band offset (the key feature), |offset|,
  along-position, tangent angle, cell-orientation-vs-axis, curvature.
- **nbr** — mean/min kNN distance, count within 2·D, neighbour-offset
  **anisotropy** (single-file IHC row ≈1, OHC band lower), mean offset of
  the k neighbours.
- **edge** — distance to the convex hull of all centroids (a
  no-annotation tissue-edge proxy; secondary).

Every cell also carries a **`flag`** bitmask (off-axis outlier, sparse
neighbourhood, cochlear-end extrapolation, too-few-cells / fold) — emit
for human review; flagged regions are exactly where fusion earns its keep.

## Modes

- **`rule`** — *zero training step*: per image, a 2-component GMM on the
  signed offset; the minority component is IHC (~1:3 prior). Fully
  deterministic. **Held-out test bal_acc 0.873** with no fit anywhere —
  the design note's headline advantage, and a sanity oracle.
- **`cv` / `train`** — `logreg` (default) or `gbm`, class-balanced,
  GroupKFold **by source image** (same protocol & metrics as the CNN).
- **`fuse`** — generates *both* train signals **out-of-fold on the same
  splits**: geom refit per fold (`oof_geom`) and the **CNN retrained per
  fold** with a nested image-level sub-val for early stopping (`oof_cnn`,
  default `fuse.cnn_oof: true`). Each OOF stream is then **monotonically
  recalibrated** (isotonic by default) so probabilities match empirical
  correctness at the real ~1:3 prior. The fusion weight `w` *and* the
  fused decision threshold are picked **jointly** on OOF for bal_acc,
  frozen, then applied to the held-out test — where the deployed
  `best.pt` is rightly used (test was never seen). McNemar fused-vs-CNN.
  Slow by design: one CNN training per fold.

## Result on the Cunningham held-out test set (1427 cells)

Deployed pipeline: isotonic calibration on both CNN and geom, fused
`w = 0.40` (CNN share), fused decision threshold `0.440` — both chosen
jointly on OOF for bal_acc.

| Config | acc | bal_acc | IHC rec | OHC rec | macro-F1 |
|---|---|---|---|---|---|
| CNN alone (calibrated) | 0.955 | 0.951 | 0.943 | 0.959 | 0.941 |
| geom alone (logreg, CV 0.976 ± 0.009) | 0.964 | 0.961 | 0.957 | 0.966 | 0.952 |
| **CNN ⊕ geom (mean, calibrated, tuned)** | **0.978** | **0.977** | **0.977** | **0.978** | **0.970** |

Geom alone already beats the CNN; fusion lifts every metric. McNemar
fused-vs-CNN on the test: 55 cells fixed vs 23 lost, χ² = 12.3,
**p ≈ 4e-4** — the gain is real, not noise.

For reference, the prior uncalibrated, fixed-`th=0.5` fuser hit
bal_acc **0.981** — slightly higher, but that operating point depended on
miscalibrated CNN over-confidence happening to align with the geom
prediction at `w = 0.5` and `threshold = 0.5`. The deployed result (0.977)
trades ~0.4 pp of bal_acc for genuinely honest probabilities (CNN test
ECE **0.038 → 0.022**, ~43 % lower) and principled fusion semantics —
see *Why calibrate?* below.

**Audit / OOF-CNN validation.** An earlier version of `fuse` read the
deployed CNN's `class_prob` straight from the train segs to pick the
fusion weight — those probs are *in-sample* (the CNN was trained on those
cells). Switching to honest per-fold CNN retraining (`fuse.cnn_oof: true`,
the default) drops the train-side CNN estimate from bal_acc **0.949
(in-sample) → 0.924 (OOF)** — the ~2.5 pt of optimism the audit
predicted. The held-out test was unchanged by this fix because test was
already untouched by either model; the gain is methodological soundness
of the train-side selection, not of the headline number.

## Why calibrate?

Both base models train against a balanced-class objective (CNN balanced
sampler, geom `class_weight=balanced`), so their raw probabilities are on
a *balanced-prior* scale — not the true ~1:3 IHC:OHC prevalence. In plain
terms: when the CNN says "0.7 chance this is IHC," that 0.7 doesn't
correspond to 70 % empirical correctness, and averaging it with the geom's
0.7 in `mean` fusion isn't really a 50/50 vote — whichever model is more
overconfident dominates.

**Isotonic recalibration** is a monotonic remapping from each model's raw
probability to one that matches the empirical frequencies on data the
model didn't see. We fit it on OOF train predictions and freeze it for
test + inference. Three concrete things it accomplishes:

1. **Honest confidences.** When `class_prob_fused` reads 0.85, it now
   means roughly 85 % empirical correctness — useful for the GUI and any
   downstream consumer that thresholds on confidence.
2. **Principled `mean` fusion.** Averaging two probabilities only makes
   sense when both are on the same scale; calibration puts them there.
   `w = 0.5` actually means equal trust in CNN and geom.
3. **Meaningful decision thresholds.** After calibration the bal_acc-
   optimal cutoff is no longer accidentally at 0.5; we tune it on OOF
   (`fuse.threshold_grid`) and freeze it for inference (`th = 0.440`).

On the held-out test the CNN's ECE drops from **0.038 → 0.022** (≈ 43 %
lower). The argmax-based metrics (bal_acc, macro-F1) move only slightly
because the rank order of cells is preserved under monotone calibration —
but the probabilities themselves are now interpretable. Both calibrators
ride inside `fuse.pkl` so deployed inference applies the same monotonic
transforms automatically.

## Model choice (sweep)

`type ∈ {logreg, gbm}` × `C ∈ {0.1, 0.3, 1, 3, 10}`, 5-fold
GroupKFold-by-image: gbm 0.9788 ± 0.0144 vs best logreg 0.9768 ± 0.0093
bal_acc — **tied within noise**, gbm ~50 % higher variance. Kept
**logreg, C=1.0**: lower-variance, faster, and the only one with
interpretable coefficients (which confirmed the biology). `C` is nearly
flat for logreg; gbm ignores it (the 5 gbm rows are identical). Model
choice barely matters anyway — fusion (CNN 0.951 → fused 0.977) is the
real lever and is model-agnostic.

## Dependencies (geom-specific notes)

Same shared `requirements.txt` as the CNN side — `scikit-image`
(`regionprops`) and `scikit-learn` are both listed there. **No torch** on
this side by design — the geometric stack stays light; the small
metric/split/config helpers are intentionally duplicated from
`ihc_ohc_classifier.py` (its copy is canonical — keep in sync) so the
geom path never imports torch.

## Usage (inside the cellpose container)

```bash
# 1. build geom tables (image-level split implicit: train/ vs test/)
python3 /helpers/ihc_ohc/ihc_ohc_geom.py \
    --data_dir /data/.../traintest/train \
    --out /helpers/ihc_ohc/runs/cache/geom_train.npz \
    --preview /helpers/ihc_ohc/runs/cache/geom_train_preview.png
python3 /helpers/ihc_ohc/ihc_ohc_geom.py \
    --data_dir /data/.../traintest/test \
    --out /helpers/ihc_ohc/runs/cache/geom_test.npz

# 2. training-free baseline / cross-validate / train the learned model
python3 /helpers/ihc_ohc/ihc_ohc_geom_clf.py rule  --config /helpers/ihc_ohc/configs/geom.yaml
python3 /helpers/ihc_ohc/ihc_ohc_geom_clf.py cv    --config /helpers/ihc_ohc/configs/geom.yaml
python3 /helpers/ihc_ohc/ihc_ohc_geom_clf.py train --config /helpers/ihc_ohc/configs/geom.yaml

# 3. write the CNN's class_prob into the segs (prereq for fusion), then fuse
python3 /helpers/ihc_ohc/ihc_ohc_classifier.py predict \
    --ckpt /helpers/ihc_ohc/runs/<stamp>_train-cnn/best.pt \
    --seg <each _seg.npy> --write
python3 /helpers/ihc_ohc/ihc_ohc_geom_clf.py fuse  --config /helpers/ihc_ohc/configs/geom.yaml

# 4. score one seg, write geom + fused decisions back (non-destructive)
python3 /helpers/ihc_ohc/ihc_ohc_geom_clf.py predict \
    --geom_ckpt /helpers/ihc_ohc/runs/<stamp>_train-geom/geom_best.pkl \
    --fuse_ckpt /helpers/ihc_ohc/runs/<stamp>_fuse/fuse.pkl --fuse \
    --seg /data/.../000_..._seg.npy --write
```

`predict` writes `class_map_geom` / `class_prob_geom` / `geom_flag` (and,
with `--fuse`, `class_map_fused` / `class_prob_fused`) to the
**prediction sidecar** `<stem>_pred.npy` — the dataset seg is never
touched. See *Prediction sidecars* above.

## Knobs

Geom builder (`ihc_ohc_geom.py`, CLI):

| Flag | Default | Note |
|---|---|---|
| `--k_neighbors` | 6 | k for the neighbour-graph features |
| `--include-augmented` | off | also use augment.py's D4 copies |
| `--preview` | — | per-image QC overlay PNG |

Geom classifier / fusion (`configs/geom.yaml`):

| Key | Default | Note |
|---|---|---|
| `model.type` | logreg | `logreg` \| `gbm` |
| `model.class_weight` | balanced | don't sacrifice the ~1:3 minority IHC |
| `model.k_neighbors` | 6 | **must match** the builder |
| `fuse.method` | mean | `mean` (1 CV-picked weight) \| `stack` (logreg) |
| `fuse.geom_source` | model | `model` (learned, OOF) \| `rule` (training-free) |
| `fuse.cnn_oof` | true | retrain CNN per fold for honest train probs |
| `fuse.calibrate` | isotonic | `isotonic` \| `platt` \| `none` — monotonic recalibrator fit on OOF |
| `fuse.threshold_grid` | 0.30–0.70 | candidate decision thresholds for the fused score, jointly tuned with `w` on OOF |
| `cv.folds` / `cv.val_frac` | 5 / 0.2 | GroupKFold by image |
| `sweep.<model key>` | — | list of candidates → CV-scored grid |

## End-to-end orchestrator

`ihc_ohc_pipeline.py` is a thin driver over the CLIs above (each module
stays the single source of truth; it only adds the batched CNN
write-back and the plot).

```bash
DATA=/data/to_zip/hcat-data/Confocal/Cunningham/traintest

# A. train everything (crops→CNN→geom table→geom→CNN write-back→fuse).
#    All three steps share one timestamp so the run dirs sit together
#    under runs/.
python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py train \
    --train_dir $DATA/train --test_dir $DATA/test

# B. score one image (or a whole dir) with the full stack, write back.
#    Pass --cnn_ckpt / --geom_ckpt / --fuse_ckpt or set them in the
#    YAML configs (data.cnn_ckpt etc) to point at the deployed run.
python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py predict --seg $DATA/test/009_..._seg.npy
python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py predict --dir $DATA/test

# C. plot masks tinted by class (source: fused | geom | cnn | gt)
python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py plot \
    --seg $DATA/test/009_..._seg.npy --source fused
```

`predict` writes `class_map_pred`/`class_prob` (CNN),
`class_map_geom`/`class_prob_geom`/`geom_flag`, and
`class_map_fused`/`class_prob_fused` to the **prediction sidecar**
`<stem>_pred.npy` — the dataset seg stays untouched. `plot` reads
predictions from the sidecar (or, for backward compatibility, from a
seg that still carries them in-line) and renders the MYO7A image with
each mask tinted IHC (red) / OHC (blue), geometry-flagged cells ringed,
and the vs-GT accuracy in the title → PNG next to the seg. To clean up
segs from runs predating the sidecar convention, use
`migrate-preds --dir <data_dir>` (opt-in; safe — strips legacy keys
only after they are persisted in the sidecar).

## Why this over early fusion

Late fusion (Path A) doesn't perturb the tuned CNN recipe, validates
independently, and sits at bal_acc 0.977 with honest probabilities.
Concatenating geom features into `TinyHCNet`'s embedding (Path B) would
need a fresh sweep for a likely smaller marginal gain — deferred unless a
single combined model is required for deployment.

---

# In-house CLC dataset (egfp) — a second, separately-deployed model

The same stack retrained on the in-house **CLC** cochlear data
(`/data/cellpose_cc/{adult,neonate}`, egfp marker, 66 images / 20 animals
/ 9617 cells, ~23% IHC). It is a **separate** model — Cunningham's locked
model, configs and manifest are untouched. Configs:
[`configs/cnn_clc.yaml`](configs/cnn_clc.yaml),
[`configs/geom_clc.yaml`](configs/geom_clc.yaml); GUI manifest
[`clc_ihc_ohc.yaml`](clc_ihc_ohc.yaml).

**Held-out (OOF, grouped 5-fold by animal):**

| Config | bal_acc | acc | IHC rec | OHC rec | macro-F1 |
|---|---|---|---|---|---|
| CNN alone | 0.956 ± 0.032 | — | — | — | — |
| geom alone | 0.974 ± 0.008 | — | — | — | — |
| **CNN ⊕ geom (mean, calibrated)** | **0.984** | 0.986 | 0.980 | 0.988 | 0.981 |

`w = 0.40` (CNN share), fused threshold 0.400. On par with Cunningham —
the geom side (stain-agnostic cochlear-row geometry) transfers and refits
with no surprises; the CNN had to relearn the egfp appearance (the
deployed Cunningham CNN was degenerate here, predicting all-IHC). Deployed
run: `runs/20260624-003309_clc-*`.

**Is fusion actually worth it? (definitive leak-free OOF, run
`20260625-134518_clc-fuse-compare`).** The `fuse` report now emits all
three pairwise McNemars. On CLC the **geometry is the stronger single
model, not the CNN** — and fusion still beats it significantly:

| comparison (OOF, grouped by animal) | discordant pairs | χ² | p | verdict |
|---|---|---|---|---|
| CNN vs geom | geom-right 385 / CNN-right 122 | 135.4 | 3e-31 | **geom ≫ CNN** |
| fused vs geom | fused-right 68 / geom-right 11 | 39.7 | **3e-10** | **fusion > geom (real)** |
| fused vs CNN | fused-right 375 / CNN-right 55 | 236.7 | 2e-53 | fusion ≫ CNN |

So the CNN does *not* outperform — geom is ~4 pts of bal_acc ahead of it
(0.975 vs 0.934 in that run) — yet fusion fixes 68 of geom's errors while
losing only 11, a statistically real gain. Keeping the fusion is
justified, but note the asymmetry: on the deployed model's *residual*
errors (the curator's 2nd-round corrections) the CNN was right 81 % and
geom 12 %, i.e. geom (weighted 0.6) outvoted a correct CNN — which is the
slice the row-consistency post-pass below cleans up. (Per-run CNN variance
shifts these by <1 pt vs the deployed run.)

**Confidence-adaptive fusion weight — tested, rejected.** The asymmetry
above suggested giving the CNN more weight where geom is *unsure*
(`w_eff = w_lo + (w_hi−w_lo)·(1−g)`, `g = 2·|p_geom−0.5|`). A Phase-0
headroom check on the OOF predictions (`oof_preds.npz`, now saved by every
`fuse` run) confirmed the *premise* — in the `g<0.6` buckets geom accuracy
is 0.52–0.77 while the CNN is 0.84–0.92 and rescues ~85–95 % of geom's
errors — but **killed the idea on headroom**: only 2.6 % of cells have
`g<0.6` (95 % sit at `g≥0.8` where geom is 0.995). The per-cell *oracle*
ceiling (best possible geom/CNN switch) is just **+0.005 bal_acc**; the
OOF-tuned adaptive-mean gains **+0.0004** and is not significant
(McNemar p=0.12, fixed-w actually winning 14 vs 6). Not worth 2 extra
params on 20 animals — and the row-consistency post-pass already harvests
that low-confidence slice more robustly (per-image geometry, 40/68 fixed,
0 breaks). Kept fixed `w=0.40`.

Three CLC-specific differences from the Cunningham flow — each is a one-
line knob, none changes the Cunningham path:

1. **Labels = the GUI display merge, not a dataset `class_map`.** CLC segs
   carry no `class_map`/XML. The per-cell training truth is
   `class_map_user` (the curator's corrections) over `class_map_fused`
   (the reviewed prediction) — exactly `celltype.display_label_map`'s
   priority — because the labeling workflow is "predict, then correct the
   mislabels." `resolve_training_label_map` falls back to
   `fused → geom → pred` **only when no GT exists**, so Cunningham is
   unchanged. The merged labels sit at a tight 21–25% IHC per image.
2. **egfp signal is in channel 0** of the RGB-wrapped `_chNN_SV.tif`
   (channel 1 is blank — the default `--channel 1` trains on an all-black
   plane; caught at the crops preview). Build CLC crops with
   `--channel 0`; it propagates to fusion/inference via the npz `meta`.
3. **Leak-free grouping is by animal**, not image (a cochlea's 8/16/32 khz
   regions must not straddle a fold). `--group_mode clc` on the builders
   and `data.group_mode: clc` in `geom_clc.yaml` mirror
   `clc_split.parse_animal`.

Rebuild (inside the container):

```bash
DST=/data/cellpose_cc/clc_ihc_ohc_all   # 66 seg+tif+pred relative symlinks
python3 /helpers/ihc_ohc/ihc_ohc_crops.py --data_dir $DST \
    --group_mode clc --channel 0 --out /helpers/ihc_ohc/runs/cache/crops_clc.npz \
    --preview /helpers/ihc_ohc/runs/cache/crops_clc_preview.png   # EYEBALL THIS
python3 /helpers/ihc_ohc/ihc_ohc_geom.py  --data_dir $DST \
    --group_mode clc --out /helpers/ihc_ohc/runs/cache/geom_clc.npz
python3 /helpers/ihc_ohc/ihc_ohc_classifier.py cv   --config /helpers/ihc_ohc/configs/cnn_clc.yaml
python3 /helpers/ihc_ohc/ihc_ohc_geom_clf.py  cv    --config /helpers/ihc_ohc/configs/geom_clc.yaml
# deploy (all data): train CNN + geom, then fuse (OOF CNN per fold)
python3 /helpers/ihc_ohc/ihc_ohc_classifier.py train --config /helpers/ihc_ohc/configs/cnn_clc.yaml
python3 /helpers/ihc_ohc/ihc_ohc_geom_clf.py  train  --config /helpers/ihc_ohc/configs/geom_clc.yaml
python3 /helpers/ihc_ohc/ihc_ohc_geom_clf.py  fuse   --config /helpers/ihc_ohc/configs/geom_clc.yaml
```

(`runs/clc_train_driver.sh` chains the last five with one shared timestamp.
No `screen`/`tmux` in the container — launch it with `setsid nohup`.)

## Row-consistency post-pass (inference-time error correction)

A second round of curator corrections on the deployed CLC model exposed a
clean residual error mode. Of 68 corrections across 17 images: the model
was **uncertain** on its errors (median fused confidence 0.56), and **91 %
of the errors sit unambiguously in one perp band** yet got the other label
— i.e. a cell plainly in the OHC band labeled IHC. On those cells raw
per-image perp geometry agreed with the curator 100 %, the CNN 81 %, but
the global geom *classifier* only 12 % (and, weighted 0.6 in fusion, it
dragged the fused call wrong). About half the errors are within 15 % of a
cochlear end (axis extrapolation); the dominant direction is OHC→IHC
mislabels (40 of 68).

So a per-image **row-consistency** pass fixes them: re-anchor the two perp
bands on the *confident* fused calls, then flip any **low-confidence**
fused label whose perp position clearly belongs to the other band. It runs
**after fusion, inside `predict_seg_geom`** (not in the model) —
`perp_signed` is already computed, the trained model is untouched, and the
params ride in `fuse.pkl` so deployed + GUI inference apply the same rule.
`row_consistency_refine()` is confidence-gated, so it never overrides a
confident call; overridden cells are recorded in the `row_override`
sidecar key for audit.

Validated on the 68 curator corrections (config `fuse.row_consistency`):

| operating point | fixes (of 68) | cells flipped wrong (of 9636) |
|---|---|---|
| **conservative** — flip only if fused conf < 0.60 (default) | **40 (59 %)** | **0** |
| balanced — < 0.75 | 55 (81 %) | 3 |

Deployed CLC config ships the conservative 0-break point
(`conf_anchor 0.85 / conf_override 0.60 / k_anchor 15`). Tune in
`geom_clc.yaml`; **off by default** in `DEFAULT_CONFIG` so Cunningham and
any other dataset are unaffected unless they opt in. The remaining ~28
higher-confidence errors are not safely auto-correctable and stay manual.

## Band guard (when the geom model's premise does not hold)

Added 2026-08-15. The geom classifier's entire signal is the **residual across
a fitted cochlear centreline** — IHC on one side of the band, OHC on the other.
That premise silently fails when the field of view spans more cochlear arc than
a low-degree polynomial can follow, and then every axis feature is noise which
the fusion happily inherits.

Measured over the 31 CLC images of the 2026-07/08 batches, centreline residual
in cell-diameter units (`band_fit_quality`):

| image class | `rms/D` |
|---|---|
| normal 63x (n=25) | 1.10 – 2.33 (median 1.49) |
| degeneration 63x (n=6) | 0.48 – 2.32 (median 1.13) |
| **20x acquisition (n=1)** | **15.64** |

`band_model_applies()` (in `ihc_ohc_geom.py`, threshold `BAND_RMS_MAX_OVER_D
= 4.0` — in the empty gap, 1.7× above the worst 63x seen and 3.9× below the
20x) gates the fusion in `_score_seg_geom`: when it fails, the fused decision
falls back to the **CNN alone at its own 0.5 operating point** (not the fuse
threshold, which was tuned for the CNN+geom mixture), and the row-consistency
pass is skipped too — it re-anchors on the same bands just declared unusable.

Validated on a 1262-cell 20x image with 1058 hand-labelled cells:

| | accuracy on the 1058 labelled cells |
|---|---|
| deployed fusion (geom included) | **0.196** |
| with the band guard (CNN alone) | **0.965** |
| CNN alone, raw | 0.988 |

and **zero** change in domain — two 63x images re-scored end-to-end returned
138/138 and 147/147 identical labels. Disable with `fuse.band_guard.enabled:
false`; tune with `max_rms_over_d`. Guard is **on by default**, including for
fuse ckpts saved before it existed.

**This is why 20x needs no separate classifier** (unlike segmentation, where
20x genuinely does): the CNN already transfers across magnification — it reads
a local crop — and it was only the geom's band model that broke. Note the
guard does **not** fire on the degeneration images: their centrelines fit fine,
and geom is not uniformly at fault there (on `8483 8khz` geom scores 0.86 while
the CNN scores 0.11). That failure mode is separate and unsolved.

## Hair-cell mask post-processing (off-band false-positive reject)

The row-consistency pass fixes *classification* errors; this pass fixes
*segmentation* false positives. On a **new** image, Cellpose sometimes
segments masks off in a neighbouring structure — a strip of eGFP+
supporting cells, debris in the lumen — that are not hair cells. These sit
far from the single continuous organ-of-Corti band, either as isolated
stragglers or as a coherent **satellite cluster**. The user's phrasing:
"if masks are labelled far from the main rows of cells, delete them."

**Mechanism — connected components, not local density.** A per-cell kNN
isolation test catches stragglers but misses a cluster (each member has
close neighbours *within* the cluster); a PCA-perp test is worse still (the
cluster contaminates the centreline fit and pulls it toward itself, so the
off-band cells read as on-band). The robust discriminator is graph
connectivity: link cells whose centroids are within `eps`·D (D = per-image
median cell diameter), and the band — whose ~4 rows are ≈1 D apart — fuses
into one giant connected component while every off-band structure, ≳10 D
away, is a separate component. `band_outlier_reject` (in `ihc_ohc_geom.py`,
torch-free) rejects a satellite component when it is both **far**
(≥`min_gap`·D from the band) and a **clear minority** (≤`max_frac`× the band
size — so a real second band segment split off by an imaging gap is never
deleted). Self-anchoring: the band defines itself as the giant component, so
false positives can't move the reference.

**Operating point, validated like row-consistency.** `clc_reject_eval.py`
runs the pass over the 67-image CLC set. Two reads:

| | gap3 | **gap5 (default)** | gap8 |
|---|---|---|---|
| false deletions on curated GT (of 9757 real cells) | 652 (6.7 %) | **3 (0.03 %)** | 3 (0.03 %) |
| TP-loss on 9237 model detections | 527 (5.7 %) | **0** | 0 |

There is a hard safety **cliff at gap 3→4**: at 3 D a real detection-gap in
the band splits it and the smaller piece is dropped (e.g. neonate sample 2's
39-cell segment); at ≥4 D that never happens. The deployed default is
`eps 2.5 / min_gap 5 / max_frac 0.4` — comfortably past the cliff, 0 real
detections lost, and it removes the full 18-cell eGFP-tissue cluster + a
straggler on the `63x 3L 16khz` deploy test image.

**Scope / honest ceiling.** In-sample FP-catch is ≈0 (0–1 of 327 model FPs):
on training-domain images the fine-tuned model already produces almost no
*distant* off-band FPs (its residual FPs are near-band over-segmentations,
which this pass deliberately does **not** touch — see the FN/merging
analysis). The value is a **deployment safety net for new images** where the
model over-segments surrounding tissue.

**GUI + sidecar, Option A.** Flags live in the `_pred.npy` sidecar
(`mask_reject` {cid→dist-to-band}, `mask_reject_applied` bool) — the dataset
seg is never mutated. The GUI's **"hair cell post-processing"** panel (above
the cell-type classifier) is **standalone**: *run* is enabled for any 2D
masks — no cell-type manifest needed — and flags off-band masks with the
locked defaults; if a cell-type manifest with a `hair_cell_postprocess:`
block *is* selected, its params override those defaults. The *apply*/*disable*
toggle **hides** flagged masks non-destructively (zeroes their alpha in
`draw_layer`; ids preserved, no `remove_cell` renumber) and can restore them;
the reject set (and applied state) round-trips through the sidecar so a
re-opened image comes back flagged/hidden. When applied, the IHC/OHC
classifier **excludes** the rejected masks (`active_reject_ids` →
`exclude_ids`): they neither skew the centreline/kNN geometry nor receive a
label (Option A). Aggregate analyses honour `mask_reject` only when
`mask_reject_applied`. The pipeline/config side stays **off by default**
(Cunningham unaffected); the manifest block is opt-in param tuning only.

---

# First held-out test on unseen animals (2026-08-15)

The 2026-07/08 batches brought 31 new 63x images from **9 animals never in
training** — the first chance to test the deployed classifier outside its own
OOF. Two numbers, and the gap between them is the point:

| | images | cells | acc | bal_acc |
|---|---|---|---|---|
| **unbiased** (near-fully-labelled images only) | 3 | 52 | **0.519** | 0.519 |
| assumed-reviewed, normal 63x | 25 | 3324 | 0.987 | 0.977 |
| assumed-reviewed, degeneration | 6 | 134 | 0.754 | 0.855 |
| assumed-reviewed, 20x | 1 | 1262 | 0.326 | 0.558 |

The 0.977 is an **upper bound, not a measurement**: only 102 of those 3324
cells were human-checked, and the rest are "correct" by the 2026-08-14 decision
to treat zero-correction images as reviewed (recorded per image, with risk
flags, in `clc_review_status_20260814.json`). The one genuinely unbiased number
available — 0.519 on the degeneration phenotype — is bad, and matches the
segmentation side, where the same cochleae score 0.177 AP.

**The structural problem this exposes:** CLC labels are the model's own accepted
output plus corrections, so on any cell a human did not touch, "truth" *is* the
deployed prediction and the model scores 100% by construction. Roughly 75 % of
the training pool is self-graded. The fix the pipeline has always needed is a
**gold set** — 1–2 animals labelled exhaustively by hand, independent of any
model. Today the project has ~63 such cells, all on degeneration images.

# Retraining on the 2026-07/08 labels — measured improvement (2026-08-15)

Combined set `clc_ihc_ohc_all2` = 97 images / 29 animals / **13 068 cells**
(vs 66 / 20 / 9 617 deployed): +165 human-labelled cells and ~3 286 accepted
model labels. Caches `crops_clc2.npz` / `geom_clc2.npz`, configs
`cnn_clc2.yaml` / `geom_clc2.yaml`.

**How to compare, given self-graded labels.** Only the 165 human-labelled cells
on the new images have truth independent of both models — everywhere else the
"truth" is the deployed model's own output, so it scores 100 % by construction
and any comparison is vacuous. Those 165 are also *error-enriched* (people
correct what's broken), so neither number below is an accuracy; the **contrast**
is what's meaningful. Splits are animal-grouped, so each cell is scored by a
model that never saw its cochlea.

| on the 165 independent cells | acc | bal_acc |
|---|---|---|
| deployed `clc-ihc-ohc` | 0.545 | 0.478 |
| retrained, run 1 | 0.709 | 0.658 |
| retrained, run 2 (replicate) | 0.721 | 0.687 |

McNemar, retrained-only-right vs deployed-only-right: **35 v 8 (p=4e-05)** and
**36 v 7 (p=9e-06)**. Replicate spread 0.012 acc — the +0.164 effect is ~13× it.

**Two confounds ruled out:**
- *Operating point.* The retrained fusion picks w=0.30/th=0.300 vs the deployed
  0.40/0.400. Sidecars store `class_prob_fused` as confidence + label, so the
  deployed model can be re-scored at any threshold: 0.50→0.545, 0.40→0.552,
  0.30→0.509, 0.20→0.467. Re-thresholding makes it *worse*; its ceiling is
  0.552. The gain is the data, not the threshold.
- *Regression on the original domain.* Old-66 OOF bal_acc, scored exactly as the
  deployed 0.984 was: **0.9850 / 0.9847** across the two runs. No regression.

**What is still unverifiable:** whether the retrained model introduced errors on
the ~3 286 cells whose labels are the deployed model's own output. That needs
the gold set (above), not another retrain.

Component ranking holds on the larger set: geom alone 0.965 > CNN alone 0.933,
fusion beats both (vs geom p=7.6e-06).

## Deployed as `clc-ihc-ohc-v2` (run `20260815-172440`)

Driver `runs/clc2_train_driver.sh` (the standalone CNN-CV step is skipped — the
fuse step already yields those OOF numbers). Artifacts:

```
runs/20260815-172440_clc2-train-cnn/best.pt
runs/20260815-172440_clc2-train-geom/geom_best.pkl
runs/20260815-172440_clc2-fuse/fuse.pkl        # w=0.30 CNN share, th=0.300
```

Manifest **`clc_ihc_ohc_v2.yaml`** (`name: clc-ihc-ohc-v2`). **v1 is untouched**
— `clc_ihc_ohc.yaml` and the `20260624-003309_*` run dirs are intact, so rolling
back is just re-selecting the v1 manifest in the GUI dropdown.

End-to-end verification: on the **20x image, which is excluded from v2's
training set and therefore genuinely held out**, v2 scores **0.974** on its 1058
hand-labelled cells (v1: 0.196; v1 + band guard: 0.965). Re-scoring images that
*are* in v2's training set (BL, CL) only confirms the checkpoints load — those
numbers are in-sample and must not be read as accuracy.

The operating point moved from w=0.40/th=0.400 to w=0.30/th=0.300 — the fuse
step's own OOF selection on the larger set, consistent with geom again
outscoring the CNN. It is not the source of the gain (see the threshold ablation
above).

# Evaluation caveats & known limitations

A review of the pipeline (June 2026) found the train/val **leakage hygiene
sound** — splits are leak-free by animal (`group_split` / `GroupKFold` on
the animal-id groups), CNN norm stats are computed on the train fold only,
the fusion OOF retrains per fold with a nested sub-val carved by group, and
the geom centerline is label-free. The CV/OOF numbers are therefore
honest *generalisation* estimates. Three caveats remain, and they bear on
how much to trust the **absolute** ~0.98 (the *relative* model ranking —
geom > CNN, fused > geom — is unaffected by all three):

1. **Operating-point selection optimism (CLC).** The fusion weight `w`,
   the decision threshold, *and* the isotonic calibrators are fit on the
   OOF train, and the headline fused bal_acc (0.984–0.987) is reported on
   that **same** OOF. Cunningham keeps a truly held-out test (its more
   conservative 0.977); the CLC model deploys on all 66 images, so it has
   **no untouched test set**. The optimism is small (w/threshold are
   low-dimensional grid picks) but real and undocumented elsewhere. To get
   an unbiased read, hold out 2–3 animals as a final single-read test (or
   nest the (w, th, calibrator) selection inside the OOF).

2. **Label circularity (CLC).** CLC has no independent ground truth: the
   training labels are the curator's corrections layered over the *prior*
   model's `class_map_fused` (the GUI display merge), and the evaluation
   truth is that same merged set. So (a) the new model is partly trained to
   reproduce the previous model on uncorrected cells, and (b) absolute
   accuracy is graded against labels partly produced by a model, which
   inflates it. The second correction round (68 new errors) is direct
   evidence of residual label noise. The only clean fix is a small
   **fully hand-labelled gold set** (1–2 animals, no model in the loop) for
   an unbiased absolute number; Cunningham's VOC-XML GT is independent and
   not subject to this.

3. **In-sample calibration / ECE.** The calibrators are fit on the OOF
   preds and ECE is reported on the same data, so the printed
   `ECE → 0.0000` is an in-sample figure, **not** a generalisation measure
   (the deployed calibrator itself is fine — it is applied to fresh data).
   Don't quote that ECE as evidence of calibration quality.

**Realistic bottom line:** the honest absolute accuracy is probably a touch
below the reported ~0.98, still strong; the deployed stack and every
*comparative* conclusion in this doc stand.

Directory inference is **batched**: `predict_seg_geom_dir` (and the CNN's
`cnn_predict_dir`) load each model once and score every seg in-process —
`ihc_ohc_pipeline.py predict --dir` no longer spawns a subprocess per seg
(≈15–20× faster on a full directory).

---

# Cellpose GUI integration

A "cell-type classifier" button now sits in the Cellpose GUI's left
sidebar (commits `c6ee093` + `75d894b`). Pick a registered manifest from
the dropdown, click **run**, and every mask gets tinted by its predicted
class — no shell, no notebook, no editing the seg file by hand.

## Manifest

The GUI doesn't hardcode the IHC/OHC stack; it loads a small yaml
**manifest** that lists which checkpoints to use. The deployed one ships
with the helpers: [`cunningham_ihc_ohc.yaml`](cunningham_ihc_ohc.yaml).

```yaml
name: cunningham-ihc-ohc                              # label in the dropdown
cnn_ckpt:  runs/20260519-114530_train-cnn/best.pt    # required
geom_ckpt: runs/20260519-145037_train-geom/geom_best.pkl  # optional
fuse_ckpt: runs/20260519-165125_fuse/fuse.pkl        # optional (needs geom_ckpt)
classes:                                              # per-class mask tint (RGB 0–255)
  IHC: [242, 64, 51]
  OHC: [38, 166, 242]
```

Relative ckpt paths are resolved against the manifest's directory, so a
`runs/<stamp>_…/` dir can carry its own manifest and stay portable.

## How it's wired

Three new pieces in [`cellpose/gui/`](../../cellpose/gui/) glue the
helpers to the GUI in-process (no subprocess, no extra container hop):

| File | Role |
|---|---|
| [`celltype.py`](../../cellpose/gui/celltype.py) | Manifest I/O + registry (`~/.cellpose/gui_celltype_models.txt`); `run_celltype(manifest, seg_path)` lazy-imports `ihc_ohc_classifier`/`ihc_ohc_geom_clf` and writes the usual sidecar. |
| [`gui.py`](../../cellpose/gui/gui.py) | Adds the "cell-type classifier" QGroupBox (dropdown + run button) and the `compute_celltype` handler that drives it. |
| [`menus.py`](../../cellpose/gui/menus.py) | Adds **Models → Add cell-type classifier (manifest .yaml)** and **Remove selected …** entries. |

`celltype.run_celltype` walks the priority `class_map_fused →
class_map_geom → class_map_pred` so whichever level of the stack is
present in the manifest is what colours the masks. The sidecar is wiped
before each run so a mask renumbering between sessions can't leave
orphan cell IDs behind.

## Using it

```text
Models menu → Add cell-type classifier (manifest .yaml)
   → pick /helpers/ihc_ohc/cunningham_ihc_ohc.yaml
sidebar "cell-type classifier" combo → choose 'cunningham-ihc-ohc'
   → click run
```

The button only enables when (a) ≥1 manifest is registered and (b) the
GUI has masks loaded. Each click:

1. Verifies the on-disk `<stem>_seg.npy` matches the GUI's current
   masks. If they differ, the click is refused and the user is prompted
   to `Ctrl+S` first (with a **WARNING** if the seg contains a
   ground-truth `class_map` — `io._save_sets` would silently drop it).
2. Wipes any stale `<stem>_pred.npy` sidecar.
3. Runs `predict_seg(cnn_ckpt, seg)` → and if `geom_ckpt` is set,
   `predict_seg_geom(geom_ckpt, seg, fuse_ckpt=…)`.
4. Reads the resulting sidecar via `load_pred()` and applies the
   manifest's per-class RGB tint to each mask.

## Constraints / gotchas

- **2D only.** The classifier indexes `seg["masks"]` as a 2-D array; the
  GUI refuses to run if `NZ != 1` (commit `75d894b`).
- **PyYAML is required when the user actually uses celltype.** Import is
  lazy so cellpose GUI environments without it still start; install via
  the [`requirements.txt`](requirements.txt) before clicking *run*.
- **Containerised launch.** `celltype.py` adds `/helpers/ihc_ohc` to
  `sys.path` on first use, so the GUI process needs to be running inside
  (or with that path mounted into) the cellpose container — same as the
  rest of this pipeline.

---

# Adding more training data

The training pipeline reads three sources for per-cell labels
(`resolve_training_label_map`, see *[Labels in / labels out](#labels-in--labels-out)*).
You almost certainly want the third — **`class_map_user` in the
`<stem>_pred.npy` sidecar** — because it's what the GUI's
hand-correction tool writes, so adding training data is a closed loop
through the GUI with no scripts to author.

## The active-learning loop (recommended)

This is the workflow the neonate set in
`/media/DATA/Chris/cellpose2D/cellpose_cc/neonate/` was built with — and
the same idea generalises to any new tissue / age / stain.

1. **Segment with Cellpose-SAM** to produce `<stem>_seg.npy` (+ paired
   `.tif`) per cochlea. No `class_map` in the seg is fine — the
   pipeline supports starting from nothing.

2. **Run the deployed celltype classifier** to seed predictions. Two
   options:

   - In the GUI: register
     [`cunningham_ihc_ohc.yaml`](cunningham_ihc_ohc.yaml) (see
     *[Cellpose GUI integration](#cellpose-gui-integration)*), open
     each seg, click *run*. Writes `class_map_pred / class_map_geom /
     class_map_fused` + probabilities into the sidecar.
   - Or batch on the CLI:

     ```bash
     podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py predict \
         --dir /data/path/to/new_cochleae/ \
         --cnn_ckpt  /helpers/ihc_ohc/runs/20260519-114530_train-cnn/best.pt \
         --geom_ckpt /helpers/ihc_ohc/runs/20260519-145037_train-geom/geom_best.pkl \
         --fuse_ckpt /helpers/ihc_ohc/runs/20260519-165125_fuse/fuse.pkl
     ```

3. **Correct in the GUI.** Click or region-select the cells the model
   got wrong; the GUI calls `set_user_labels_bulk` which writes
   `class_map_user` into the sidecar. You don't have to relabel every
   cell — only the ones you disagree with. (For the neonate set today,
   `class_map_user` is partial: 18–101 OHC corrections per cochlea, out
   of ~143–167 total masks. That's fine: only labeled cells contribute
   to training, the rest are dropped at crop/geom-build time.)

   `class_map_user` is **persistent** — re-running `predict` snapshots
   it first and re-applies it on top of the fresh predictions, so user
   work never gets overwritten by a model re-run.

4. **Decide what counts as "in the train set."** Three rough heuristics:

   - **High-trust corrections:** clear, unambiguous cells the user
     hand-clicked are real GT. Use them.
   - **Watch the class balance.** If the user has only labeled one
     class (e.g. the neonate sidecars are all `OHC`-only — see
     *Caveats* below), training on that subset alone biases the
     classifier. Either label the other class too, or weight that
     cochlea's contribution down via `--include-augmented` / not.
   - **Skip `geom_flag > 0` cells** that the user didn't label — those
     are the geom's "I'm uncertain" flag, so model labels on them are
     least trustworthy and shouldn't be promoted without a human.

5. **Drop the dir into `train_dir/` (or a sibling dir) and rebuild the
   caches.** The crop/geom builders walk every `*_seg.npy` under
   `--data_dir`, call `resolve_training_label_map`, and bake the merged
   labels into the npz — no further wiring needed:

   ```bash
   # crops .npz now includes the labeled cells from the new cochleae too
   podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_crops.py \
       --data_dir /data/path/to/combined_train/ \
       --out     /helpers/ihc_ohc/runs/cache/crops_train.npz \
       --preview /helpers/ihc_ohc/runs/cache/crops_train_preview.png
   # same for geom_train.npz with ihc_ohc_geom.py
   ```

   The per-image log lines show the provenance tag — look for `[user]`
   or `[seg+user]` to confirm new cochleae actually contributed cells.

6. **Retrain.** Run the full pipeline (`ihc_ohc_pipeline.py train …`) or
   the individual `train`/`fuse` CLIs — see *[Reproducible
   runbook](#reproducible-runbook--what-produced-the-deployed-0981)*
   below. Each step writes to a fresh timestamped run dir; the old
   deployed models stay put until you update the YAML to point at the
   new ones.

## Path B — annotated VOC-XML bounding boxes (the original Cunningham recipe)

If you have hand-drawn boxes in PASCAL VOC XML form (the format the
upstream HCAT-data ships), use that path instead — it produces a dense
`class_map` covering every mask:

```bash
# writes class_map into every _seg.npy in data_dir, based on max-overlap
# with the matching <base>.xml. See helpers/label_xfer.py.
podman exec cellpose python3 /helpers/label_xfer.py \
    --data_dir /data/path/to/new_cochleae/
```

After that, the cochleae become indistinguishable from Cunningham as far
as training is concerned — `seg["class_map"]` is the source, no user
sidecar needed.

## Caveats with the current `class_map_user` data

A quick survey of the neonate sidecars
(`/media/DATA/Chris/cellpose2D/cellpose_cc/neonate/`) shows two patterns
worth flagging before retraining:

- **All `class_map_user` entries to date are `"OHC"`** — the user has
  been correcting *only* the cells the model mislabeled as IHC. That's
  the right active-learning instinct, but it means:
  - the labeled subset has *no* positive IHC examples per neonate
    cochlea
  - the CNN's `sampler: balanced` will still try to give it ~50/50
    batches drawn from the *whole* pool (Cunningham + neonate), so IHC
    examples come from Cunningham only — fine for now, but worth a
    second pass to label some clear IHC for shape coverage at the
    neonate end of the manifold.
- **Coverage is variable** (18 cells on one cochlea, 101 on another).
  Cochleae with very few user labels contribute very few cells to
  training. Check the per-image build log after the next
  `ihc_ohc_crops.py` run to see who pulls weight.

## Sanity checks before retraining

- `iter_seg_files(train_dir)` should report the new cochleae alongside
  the old:

  ```bash
  podman exec cellpose python3 -c "
  import sys; sys.path.insert(0, '/helpers/ihc_ohc')
  from ihc_ohc_crops import iter_seg_files
  print(sum(1 for _ in iter_seg_files('/data/.../combined_train')), 'cochleae')"
  ```

- The leading numeric prefix (`group_key` in
  [`ihc_ohc_crops.py`](ihc_ohc_crops.py)) must be **disjoint** between
  `train/` and `test/` or splits will leak. If you're adding cochleae,
  bump the prefix past the highest existing one (`ls train/ | sort -n |
  tail`). For the neonate files (no numeric prefix), `group_key` falls
  back to the aug/channel-stripped stem — fine, but check by-hand that
  the same cochlea doesn't appear in both dirs.

- Eyeball `runs/cache/crops_train_preview.png` after rebuilding. Strongly
  skewed counts (e.g. 95 % OHC after adding the neonate set) usually
  means the user-label distribution is what's driving it — see *Caveats*
  above.

---

# Reproducible runbook — what produced the deployed 0.977

This is the exact path that yielded the legacy artifacts now preserved
in `runs/20260519-*_…/`. All commands run inside the cellpose container
(see [Run everything inside the container](../../../CLAUDE.md) in the
project README for the mount layout); `$DATA` is
`/data/to_zip/hcat-data/Confocal/Cunningham/traintest`.

### 0. One-time setup

```bash
# Container deps not in stock cellpose
podman exec cellpose pip install -r /helpers/ihc_ohc/requirements.txt
```

### 1. Build crops (CNN input)

```bash
podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_crops.py \
    --data_dir $DATA/train \
    --out     /helpers/ihc_ohc/runs/cache/crops_train.npz \
    --preview /helpers/ihc_ohc/runs/cache/crops_train_preview.png

podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_crops.py \
    --data_dir $DATA/test \
    --out     /helpers/ihc_ohc/runs/cache/crops_test.npz
```

→ 93 train cochleae / 23 test cochleae, 1427 test cells. Eyeball the
preview PNG before going further.

### 2. Build the geom feature table (geom input)

```bash
podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_geom.py \
    --data_dir $DATA/train \
    --out     /helpers/ihc_ohc/runs/cache/geom_train.npz \
    --preview /helpers/ihc_ohc/runs/cache/geom_train_preview.png

podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_geom.py \
    --data_dir $DATA/test \
    --out     /helpers/ihc_ohc/runs/cache/geom_test.npz
```

### 3. (Optional) sweep + cross-validate

The defaults in `configs/cnn.yaml` and `configs/geom.yaml` *are* the
sweep winners — only repeat this if you've changed the data
substantially. Edit the `sweep:` block in the relevant yaml first, then:

```bash
podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_classifier.py sweep \
    --config /helpers/ihc_ohc/configs/cnn.yaml
podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_classifier.py cv \
    --config /helpers/ihc_ohc/configs/cnn.yaml
```

For the deployed model: `lr=0.002, loss=ce, select_ema=0.5,
class_weight=none, sampler=balanced` (CNN); `type=logreg,
class_weight=balanced, C=1.0, k_neighbors=6` (geom). Both are the
current defaults — sweep history in *Model choice* below.

### 4. Train, in a screen session

The orchestrator threads one timestamp across all three steps so the run
dirs cluster. Use screen so you can detach and monitor:

```bash
screen -dmS ihc_train -L -Logfile /tmp/ihc_train.log bash -c "
podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py train \
    --train_dir $DATA/train \
    --test_dir  $DATA/test \
    --cnn_config  /helpers/ihc_ohc/configs/cnn.yaml \
    --geom_config /helpers/ihc_ohc/configs/geom.yaml
"
# attach to watch:   screen -r ihc_train
# detach again:      Ctrl-A  d
```

Wall-clock on a 2060 SUPER: ~5 min CNN train + ~20 s geom + ~25 min
fusion (the OOF CNN retrains 5 CNNs, one per fold — that's the bulk).

The orchestrator prints three sibling run dirs at the end, all sharing
one timestamp:

```
✓ pipeline complete:
   CNN  ckpt → /helpers/ihc_ohc/runs/<stamp>_train-cnn/best.pt
   geom ckpt → /helpers/ihc_ohc/runs/<stamp>_train-geom/geom_best.pkl
   fuse ckpt → /helpers/ihc_ohc/runs/<stamp>_fuse/fuse.pkl
```

Each dir also contains a frozen `config.yaml`, the resolved input config
exactly as the run saw it.

### 5. Promote the new run to "deployed"

There's no symlink — `data.cnn_ckpt` / `geom_ckpt` / `fuse_ckpt` in the
two YAML configs is the single source of truth for "what `predict`
loads." Edit them to point at the three paths the pipeline just printed:

```yaml
# configs/cnn.yaml
data:
  cnn_ckpt: /helpers/ihc_ohc/runs/<new-stamp>_train-cnn/best.pt

# configs/geom.yaml
data:
  geom_ckpt: /helpers/ihc_ohc/runs/<new-stamp>_train-geom/geom_best.pkl
  fuse_ckpt: /helpers/ihc_ohc/runs/<new-stamp>_fuse/fuse.pkl
```

Also update [`cunningham_ihc_ohc.yaml`](cunningham_ihc_ohc.yaml) (the
GUI manifest) to point at the new dirs if you want the GUI to pick up
the new model.

### 6. Confirm against the held-out test

```bash
podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py predict \
    --dir $DATA/test
```

The per-image `vs GT [seg]: acc …` lines and the McNemar test printed by
`fuse` are the headline. The locked target for the Cunningham set is
**0.977 fused bal_acc / 0.970 macro-F1** — anything materially below
that on the same data is a regression, treat the new run as a candidate
not a replacement.
