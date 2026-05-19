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
