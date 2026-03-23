"""
Convert VOC bounding-box annotations → SAM masks → Cellpose _seg.npy

Pipeline per image:
  1. Load multi-channel TIF, extract MYO7A channel (index 1, 0-based)
  2. Normalize uint16 → uint8 and replicate to 3-ch RGB for SAM
  3. Run SAM with each bounding box as a prompt
  4. Combine instance masks into a labelled array
  5. Save:
       <stem>_myo7a.tif       — single-channel uint16 image for cellpose training
       <stem>_myo7a_seg.npy   — cellpose seg dict with 'masks' key
"""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
from segment_anything import SamPredictor, sam_model_registry

# cellpose lives in a sibling git checkout; add it to path if needed
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cellpose_git"))
from cellpose.utils import masks_to_outlines

# ── SAM setup ──────────────────────────────────────────────────────────────────
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MYO7A_CHANNEL = 1          # 0-indexed; channel 2 in the .txt files
DATA_DIR = "/media/DATA/Chris/cellpose2D/to_zip/hcat-data/Confocal/Cunningham"
OUT_DIR = DATA_DIR         # save alongside source files (change if preferred)

sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to("cuda")
predictor = SamPredictor(sam)


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path):
    """Return (boxes, classes) from a VOC-style XML.
    boxes: list of [xmin, ymin, xmax, ymax] ints
    classes: list of str ('IHC' or 'OHC')
    """
    root = ET.parse(xml_path).getroot()
    boxes, classes = [], []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bndbox  = obj.find("bndbox")
        if name_el is None or bndbox is None:
            continue
        name = name_el.text or ""
        xmin_el = bndbox.find("xmin")
        ymin_el = bndbox.find("ymin")
        xmax_el = bndbox.find("xmax")
        ymax_el = bndbox.find("ymax")
        if any(el is None for el in (xmin_el, ymin_el, xmax_el, ymax_el)):
            continue
        box = [
            int(xmin_el.text),  # type: ignore[arg-type]
            int(ymin_el.text),  # type: ignore[arg-type]
            int(xmax_el.text),  # type: ignore[arg-type]
            int(ymax_el.text),  # type: ignore[arg-type]
        ]
        w, h = box[2] - box[0], box[3] - box[1]
        if w < 5 or h < 5:
            print(f"  skip degenerate box: {name} {box}")
            continue
        boxes.append(box)
        classes.append(name)
    return boxes, classes


def channel_to_uint8_rgb(channel_u16):
    """Normalize a uint16 2-D image to uint8 and replicate to (H, W, 3) for SAM."""
    lo, hi = np.percentile(channel_u16, (1, 99))
    clipped = np.clip(channel_u16, lo, hi)
    if hi > lo:
        scaled = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        scaled = np.zeros_like(channel_u16, dtype=np.uint8)
    return np.stack([scaled, scaled, scaled], axis=-1)  # (H, W, 3)


def process_image(tif_path, xml_path, out_dir):
    stem = os.path.splitext(os.path.basename(tif_path))[0]
    out_tif  = os.path.join(out_dir, stem + "_myo7a.tif")
    out_seg  = os.path.join(out_dir, stem + "_myo7a_seg.npy")

    if os.path.exists(out_seg):
        print(f"  already done, skipping")
        return

    # 1. load TIF — normalise to (C, H, W) uint16
    img_stack = tifffile.imread(tif_path)
    if img_stack.ndim != 3:
        raise ValueError(f"Expected a 3-D stack, got shape {img_stack.shape}")
    # Handle both (C, H, W) and (H, W, C) layouts
    if img_stack.shape[2] <= 4 and img_stack.shape[0] > 4:
        # channels-last (H, W, C) → (C, H, W)
        img_stack = np.moveaxis(img_stack, -1, 0)
    myo7a = img_stack[MYO7A_CHANNEL]        # (H, W) uint16
    if myo7a.max() == 0:
        # channel is blank (e.g. image 011 has a dead middle channel)
        # fall back to the brightest non-zero channel
        alt = max((img_stack[c] for c in range(img_stack.shape[0]) if c != MYO7A_CHANNEL),
                  key=lambda ch: ch.mean())
        print(f"  WARNING: channel {MYO7A_CHANNEL} is blank, falling back to brightest channel")
        myo7a = alt

    # 2. parse annotations
    boxes, classes = parse_voc_xml(xml_path)
    if not boxes:
        print(f"  no valid boxes, skipping")
        return

    # 3. set SAM image (needs uint8 RGB)
    rgb = channel_to_uint8_rgb(myo7a)       # (H, W, 3) uint8
    predictor.set_image(rgb)

    # 4. run SAM per box, build instance mask
    H, W = myo7a.shape
    mask_out  = np.zeros((H, W), dtype=np.int32)
    class_map = {}   # cell_id → class name

    for cell_id, (box, cls) in enumerate(zip(boxes, classes), start=1):
        input_box = np.array(box, dtype=float)   # shape (4,) — SAM expects 1-D
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=True,
        )
        best = masks[np.argmax(scores)]          # bool (H, W)

        # write only to unoccupied pixels (no overlap)
        free = mask_out == 0
        mask_out[best & free] = cell_id
        class_map[cell_id] = cls

    # 5. save MYO7A channel TIF for cellpose training
    tifffile.imwrite(out_tif, myo7a)

    # 6. save cellpose _seg.npy
    outlines = masks_to_outlines(mask_out)   # bool (H, W), required by GUI
    np.save(
        out_seg,
        {
            "masks":       mask_out,
            "outlines":    outlines,
            "filename":    out_tif,
            "img":         myo7a,
            "class_map":   class_map,   # bonus: IHC/OHC labels for later use
        },
    )
    n_cells = mask_out.max()
    print(f"  saved {n_cells} masks → {os.path.basename(out_seg)}")


# ── batch loop ─────────────────────────────────────────────────────────────────

tif_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.tif")))
print(f"Found {len(tif_paths)} TIF files")

for tif_path in tif_paths:
    xml_path = tif_path.replace(".tif", ".xml")
    if not os.path.exists(xml_path):
        print(f"No XML for {tif_path}, skipping")
        continue

    print(f"Processing {os.path.basename(tif_path)}")
    try:
        process_image(tif_path, xml_path, OUT_DIR)
    except Exception as e:
        print(f"  ERROR: {e}")

print("Done.")
