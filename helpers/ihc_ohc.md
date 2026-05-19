# IHC vs OHC Classifier

A tiny CNN that labels each Cellpose-segmented hair cell as an **inner**
(IHC) or **outer** (OHC) hair cell, from the MYO7A channel plus the
instance's own mask. Two scripts + one config:

| File | Phase | Role |
|---|---|---|
| `ihc_ohc_crops.py` | 1 — data prep / inference cropping | Expand each instance bbox, pad, add the target-mask channel, resize → fixed crops |
| `ihc_ohc_classifier.py` | 2 — model | `TinyHCNet` `sweep` / `cv` / `train` / `predict`; reuses Phase-1 cropping at inference |
| `ihc_ohc_config.yaml` | — | all classifier hyperparameters (sweep / CV / train) |

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

Beyond the cellpose env, `ihc_ohc_classifier.py` needs **PyYAML** (config)
and **scikit-learn** (`sklearn.model_selection.GroupKFold` for CV) installed
in the container. `ihc_ohc_crops.py` needs only cellpose's own deps
(numpy/scipy/cv2/tifffile); matplotlib is optional (montages degrade
gracefully if absent).

**Container note:** the cellpose container's `/dev/shm` is only 63 MB, too
small for PyTorch DataLoader worker IPC — keep `train.workers: 0` (the
default). The crops are already an in-RAM numpy array, so 0 is also fast.
Raising it bus-errors unless the container is restarted with a larger
`--shm-size`.

## Configuration

All classifier hyperparameters live in a YAML config, **not** CLI flags —
runs are reproducible and sweeps are declarative. See
[`ihc_ohc_config.yaml`](ihc_ohc_config.yaml) for the annotated schema
(`data:` / `train:` / `cv:` / `sweep:`). A partial file is valid (missing
keys fall back to `DEFAULT_CONFIG`); unknown keys are rejected; the resolved
config is echoed at the start of every run. Crop building
(`ihc_ohc_crops.py`) stays CLI-flag driven (it is run once).

## Usage

Run inside the cellpose container (`podman exec cellpose …`). Mounts:
`/data` = `/media/DATA/Chris/cellpose2D`, `/helpers` = this folder.

```bash
# 1. build crops (image-level split is implicit: train/ vs test/ dirs)
python3 /helpers/ihc_ohc_crops.py \
    --data_dir /data/to_zip/hcat-data/Confocal/Cunningham/traintest/train \
    --out /helpers/crops_train.npz --preview /helpers/crops_train_preview.png
python3 /helpers/ihc_ohc_crops.py \
    --data_dir /data/to_zip/hcat-data/Confocal/Cunningham/traintest/test \
    --out /helpers/crops_test.npz

# 2. eyeball crops_train_preview.png BEFORE training

# 3a. (optional) hyperparameter sweep — edit `sweep:` in the config first
python3 /helpers/ihc_ohc_classifier.py sweep --config /helpers/ihc_ohc_config.yaml

# 3b. (optional) cross-validate the chosen config (generalisation estimate)
python3 /helpers/ihc_ohc_classifier.py cv    --config /helpers/ihc_ohc_config.yaml

# 4. train the final model (one split → checkpoint + held-out test report)
python3 /helpers/ihc_ohc_classifier.py train --config /helpers/ihc_ohc_config.yaml

# 5. predict on a segmentation (optionally write predictions back in)
python3 /helpers/ihc_ohc_classifier.py predict \
    --ckpt /helpers/ihc_ohc_run/best.pt \
    --seg  /data/.../000_cunningham_mouse_confocal_myo7a_seg.npy --write
```

Typical flow: **sweep → pick the top row → copy its values into `train:` →
cv → train**. Outputs in `data.out_dir`: `best.pt` (weights + norm stats +
crop params + resolved `train_cfg`), `history.json`, `sweep_results.json`,
a test confusion matrix (stdout), and `test_misclassified.png`.

## Knobs

Crop builder (`ihc_ohc_crops.py`, CLI flags):

| Flag | Default | Note |
|---|---|---|
| `--out_size` | 64 | crop side |
| `--pad_frac` / `--pad_px` | 0.5 / — | context padding |
| `--pad_value` | mean | `0` for zero-pad |
| `--hard_mask` | off | binarise mask channel |
| `--include-augmented` | off | also use on-disk D4 copies |

Classifier (`ihc_ohc_config.yaml`):

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
`class_prob` into the seg dict, leaving `masks` untouched — so
`plot_boxes.py` and the Cellpose GUI still work, and the probability is
available as a confidence score to fuse with the geometric classifier
(weighted vote / tiebreaker).

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
| `ihc_ohc_geom_config.yaml` | — | all geom + fusion hyperparameters |

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
- **`fuse`** — reads the CNN's `class_prob` from the segs, generates geom
  probs **out-of-fold** (model refit per fold → honest estimate), picks
  the fusion weight on OOF, freezes it, scores the held-out test, and runs
  a **McNemar** test of fused-vs-CNN.

## Result on the Cunningham held-out test set (1427 cells)

| Config | acc | bal_acc | IHC rec | OHC rec | macro-F1 |
|---|---|---|---|---|---|
| CNN alone | 0.929 | 0.945 | 0.977 | 0.914 | 0.911 |
| geom alone (logreg, CV 0.976 ± 0.009) | 0.963 | 0.962 | 0.960 | 0.964 | 0.951 |
| **CNN ⊕ geom (mean, w=0.5)** | **0.980** | **0.981** | **0.983** | **0.980** | **0.974** |

Geom alone already beats the CNN; fusion lifts **every** metric with no
trade-off. McNemar fused-vs-CNN on the test set: 78 cells fixed vs 5 lost,
χ²=62.5, **p ≈ 3e-15** — the gain is real, not noise. The CNN's weak spot
(OHC recall 0.914) is exactly what the geometry repairs (→ 0.980).

## Model choice (sweep)

`type ∈ {logreg, gbm}` × `C ∈ {0.1, 0.3, 1, 3, 10}`, 5-fold
GroupKFold-by-image: gbm 0.9788 ± 0.0144 vs best logreg 0.9768 ± 0.0093
bal_acc — **tied within noise**, gbm ~50 % higher variance. Kept
**logreg, C=1.0**: lower-variance, faster, and the only one with
interpretable coefficients (which confirmed the biology). `C` is nearly
flat for logreg; gbm ignores it (the 5 gbm rows are identical). Model
choice barely matters anyway — fusion (0.945 → 0.981) is the real lever
and is model-agnostic.

## Dependencies

Beyond the cellpose env, `ihc_ohc_geom.py` needs **scikit-image**
(`regionprops`; newly installed in the container) + scipy (already
present); `ihc_ohc_geom_clf.py` needs scikit-learn (already required for
the CNN's `GroupKFold`). **No torch** — the geometric stack stays light;
the small metric/split/config helpers are intentionally duplicated from
`ihc_ohc_classifier.py` (its copy is canonical — keep in sync) so the geom
path never imports torch.

## Usage (inside the cellpose container)

```bash
# 1. build geom tables (image-level split implicit: train/ vs test/)
python3 /helpers/ihc_ohc_geom.py \
    --data_dir /data/.../traintest/train \
    --out /helpers/geom_train.npz --preview /helpers/geom_train_preview.png
python3 /helpers/ihc_ohc_geom.py \
    --data_dir /data/.../traintest/test --out /helpers/geom_test.npz

# 2. training-free baseline / cross-validate / train the learned model
python3 /helpers/ihc_ohc_geom_clf.py rule  --config /helpers/ihc_ohc_geom_config.yaml
python3 /helpers/ihc_ohc_geom_clf.py cv    --config /helpers/ihc_ohc_geom_config.yaml
python3 /helpers/ihc_ohc_geom_clf.py train --config /helpers/ihc_ohc_geom_config.yaml

# 3. write the CNN's class_prob into the segs (prereq for fusion), then fuse
python3 /helpers/ihc_ohc_classifier.py predict \
    --ckpt /helpers/ihc_ohc_run/best.pt --seg <each _seg.npy> --write
python3 /helpers/ihc_ohc_geom_clf.py fuse  --config /helpers/ihc_ohc_geom_config.yaml

# 4. score one seg, write geom + fused decisions back (non-destructive)
python3 /helpers/ihc_ohc_geom_clf.py predict \
    --geom_ckpt /helpers/ihc_ohc_geom_run/geom_best.pkl \
    --fuse_ckpt /helpers/ihc_ohc_geom_run/fuse.pkl --fuse \
    --seg /data/.../000_..._seg.npy --write
```

`predict` writes `class_map_geom` / `class_prob_geom` / `geom_flag` (and,
with `--fuse`, `class_map_fused`) into the seg dict, leaving `masks` and
the CNN's keys untouched — same non-destructive convention, so
`plot_boxes.py` and the GUI keep working.

## Knobs

Geom builder (`ihc_ohc_geom.py`, CLI):

| Flag | Default | Note |
|---|---|---|
| `--k_neighbors` | 6 | k for the neighbour-graph features |
| `--include-augmented` | off | also use augment.py's D4 copies |
| `--preview` | — | per-image QC overlay PNG |

Geom classifier / fusion (`ihc_ohc_geom_config.yaml`):

| Key | Default | Note |
|---|---|---|
| `model.type` | logreg | `logreg` \| `gbm` |
| `model.class_weight` | balanced | don't sacrifice the ~1:3 minority IHC |
| `model.k_neighbors` | 6 | **must match** the builder |
| `fuse.method` | mean | `mean` (1 CV-picked weight) \| `stack` (logreg) |
| `fuse.geom_source` | model | `model` (learned, OOF) \| `rule` (training-free) |
| `cv.folds` / `cv.val_frac` | 5 / 0.2 | GroupKFold by image |
| `sweep.<model key>` | — | list of candidates → CV-scored grid |

## Why this over early fusion

Late fusion (Path A) doesn't perturb the tuned CNN recipe, validates
independently, and already clears 0.98. Concatenating geom features into
`TinyHCNet`'s embedding (Path B) would need a fresh sweep for a likely
smaller marginal gain — deferred unless a single combined model is
required for deployment.
