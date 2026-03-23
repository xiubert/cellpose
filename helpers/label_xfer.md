# Bounding Box → SAM Mask → Cellpose Training Data

Converts VOC-style bounding box annotations into instance segmentation masks using SAM (Segment Anything Model), then saves them in Cellpose's `_seg.npy` format for fine-tuning.

## Motivation

Datasets such as Buswinka et al. (2024)[^1] ship with bounding box annotations but Cellpose requires pixel-level instance masks for training. SAM is well-suited to this gap: given a tight bounding box around a cell body, it reliably segments the interior. This script automates that conversion across an entire directory of images.

[^1]: Buswinka, C. J., Rosenberg, D. B., Simikyan, R. G., et al. "Large-Scale Annotated Dataset for Cochlear Hair Cell Detection and Classification." *Scientific Data* 11, 416 (2024). https://doi.org/10.1038/s41597-024-03218-y

## Pipeline

```
multi-channel TIF  →  extract target channel (0-based index)
                   →  normalize uint16 → uint8 RGB
                   →  SAM box-prompted segmentation per annotation
                   →  combine into labelled instance mask
                   →  save <stem>_ch<N>.tif + <stem>_ch<N>_seg.npy
```

## Example dataset

| Field | Value |
|---|---|
| Files | 116 sets of `.tif` / `.xml` / `.txt` |
| TIF format | 2D (single plane / MIP) — no Z dimension |
| TIF shape | `(3, 1024, 1024)` uint16 |
| Channel 1 (index 0) | Phalloidin (488 nm) |
| Channel 2 (index 1) | anti-MYO7A (568 nm) |
| Annotation format | VOC XML, classes `IHC` and `OHC` |
| Pixel size | 131.8 nm, 63× 1.4 NA oil, Zeiss LSM780 |

## Outputs

For each input `<stem>.tif`, two files are written to `OUT_DIR`:

- **`<stem>_ch<N>.tif`** — single-channel uint16 image; this is the image Cellpose trains on.
- **`<stem>_ch<N>_seg.npy`** — Cellpose seg dict with keys:
  - `masks` — int32 instance label array `(H, W)`, 0 = background
  - `outlines` — bool `(H, W)` outline array computed via `masks_to_outlines()`, required for the Cellpose GUI
  - `img` — the extracted channel array
  - `filename` — absolute path to the paired `.tif`
  - `class_map` — dict mapping cell ID → class label (e.g. `"IHC"` / `"OHC"`)

## Requirements

- Python environment with `cellpose` and `segment-anything` installed
- SAM checkpoint (e.g. `sam_vit_h_4b8939.pth`) — download from the [SAM releases page](https://github.com/facebookresearch/segment-anything#model-checkpoints)

## Usage

```bash
python label_xfer.py \
    --checkpoint sam_vit_h_4b8939.pth \
    --channel 1 \
    --data_dir /path/to/data \
    --out_dir /path/to/output      # optional; defaults to data_dir
```

Arguments can also be set via environment variables (`SAM_CHECKPOINT`, `CHANNEL`, `DATA_DIR`, `OUT_DIR`).

The script skips any file whose `_seg.npy` already exists, so it is safe to re-run after interruption. Images where the target channel is blank are skipped with a warning.

## Cellpose Training

After running the script, point Cellpose at the output directory:

```bash
python -m cellpose --train \
    --dir /path/to/output \
    --img_filter _ch1 \
    --pretrained_model cyto3 \
    --n_epochs 100
```

Adjust `--img_filter` to match the channel suffix used (e.g. `_ch0`, `_ch2`).

## Notes

- **Channel layout**: both `(C, H, W)` and `(H, W, C)` TIF layouts are auto-detected and normalized at load time.
- **Degenerate boxes**: bounding boxes smaller than 5×5 px are skipped.
- **Overlapping cells**: if two SAM masks overlap, the first cell (lower cell ID) takes priority for contested pixels.
- **Normalization**: the 1st–99th percentile of the channel is used to clip outlier pixels before converting to uint8 for SAM.
- **Outlines**: computed with `cellpose.utils.masks_to_outlines` and stored in the seg file; required for correct rendering in the Cellpose GUI.
