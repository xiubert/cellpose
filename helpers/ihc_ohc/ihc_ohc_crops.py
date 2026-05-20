"""
Phase 1 — build an IHC/OHC crop dataset from Cellpose _seg.npy files.

This is the reusable crop-expansion code: the *same* functions build the
training set here and crop cells at inference time
(see ihc_ohc_classifier.py:predict_seg).

Per cell:
  1. bbox from the instance mask           (np.where on `masks == cell_id`)
  2. expand the bbox by a fraction of its size (context around the cell)
  3. crop with zero/mean padding           (robust to image-edge overflow)
  4. add the *target instance* binary mask as a second channel
  5. resize to a fixed square (cv2)         (myo7a: area/linear, mask: linear)
  → (2, S, S) float32 crop, raw intensities (normalisation lives in the
    classifier so train and inference share one source of truth)

Where the IHC/OHC labels come from
----------------------------------
label_xfer.py / augment.py write a `class_map` {mask_id: "IHC"|"OHC"} into
every _seg.npy (verified present on the Cunningham train/test set, originals
*and* D4-augmented copies). That is the primary source. If a seg file lacks
class_map, we fall back to the original VOC XML
(<NNN>_cunningham_mouse_confocal.xml) and assign each mask the class of the
bounding box it overlaps most — the same correspondence label_xfer built.

Augmented copies
----------------
The data dir contains augment.py's D4 variants (_rot90, _rot180, _fliph …).
By default they are skipped: the classifier does its own rotation/flip
augmentation at train time, so the on-disk copies only inflate epochs and risk
train/val leakage. They are still grouped by source-image id, so
--include-augmented is safe if you want them.

Usage
-----
  # in the cellpose container; /data == /media/DATA/Chris/cellpose2D
  python /helpers/ihc_ohc_crops.py \
      --data_dir /data/to_zip/hcat-data/Confocal/Cunningham/traintest/train \
      --out /helpers/crops_train.npz --preview /helpers/crops_train_preview.png

  python /helpers/ihc_ohc_crops.py \
      --data_dir /data/to_zip/hcat-data/Confocal/Cunningham/traintest/test \
      --out /helpers/crops_test.npz
"""

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tifffile

# IHC/OHC → integer label. Fixed so train/inference always agree.
CLASS_NAMES = ["IHC", "OHC"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# D4 tags written by augment.py — used both to detect and to strip aug copies.
_AUG_TAGS = (
    "rot90_fliph", "rot90_flipv",  # compound first so prefixes don't shadow them
    "rot90", "rot180", "rot270", "fliph", "flipv",
)


# ── filename helpers ───────────────────────────────────────────────────────────

def seg_stem(seg_path):
    """'…/000_…_myo7a_fliph_seg.npy' → '000_…_myo7a_fliph'."""
    base = os.path.basename(seg_path)
    return base[:-len("_seg.npy")] if base.endswith("_seg.npy") else os.path.splitext(base)[0]


def is_augmented(stem):
    """True if the stem ends with one of augment.py's D4 tags."""
    return any(stem.endswith(f"_{t}") for t in _AUG_TAGS)


def base_stem(stem):
    """Strip the aug tag and the '_myo7a' suffix → the per-image base stem."""
    for t in _AUG_TAGS:
        if stem.endswith(f"_{t}"):
            stem = stem[: -len(f"_{t}")]
            break
    return stem[: -len("_myo7a")] if stem.endswith("_myo7a") else stem


def group_key(seg_path):
    """Source-image id for leak-free splitting.

    All augmentations of one image must share a split, so prefer the leading
    numeric prefix (e.g. '000'); fall back to the aug/channel-stripped stem.
    """
    m = re.match(r"(\d+)", os.path.basename(seg_path))
    return m.group(1) if m else base_stem(seg_stem(seg_path))


def tif_for_seg(seg_path):
    """Paired '<stem>.tif' next to the seg file (preferred image source)."""
    return os.path.join(os.path.dirname(seg_path), f"{seg_stem(seg_path)}.tif")


# ── prediction sidecar (don't mutate the dataset) ───────────────────────────────

# Every key produced downstream of segmentation: classifier (CNN), geom,
# and fusion. The sidecar is the single source of truth for these; the
# `_seg.npy` dataset files stay untouched.
PRED_KEYS = ("class_map_pred", "class_prob",
             "class_map_geom", "class_prob_geom", "geom_flag",
             "class_map_fused", "class_prob_fused")


def pred_path(seg_path):
    """'<stem>_seg.npy' → '<stem>_pred.npy' next to the seg."""
    if seg_path.endswith("_seg.npy"):
        return seg_path[:-len("_seg.npy")] + "_pred.npy"
    base, _ = os.path.splitext(seg_path)
    return base + "_pred.npy"


def load_pred(seg_path, seg=None):
    """Read the per-cell predictions for one seg.

    Prefers the sidecar; falls back transparently to keys *inside* the seg
    so segs written by earlier runs (when predictions were stored in-line)
    keep working. Pass `seg` (already-loaded dict) to skip a disk read in
    the fallback path. Returns {} if neither source has predictions.
    """
    pp = pred_path(seg_path)
    if os.path.exists(pp):
        try:
            d = np.load(pp, allow_pickle=True).item()
        except Exception:  # noqa: BLE001
            d = {}
        return d if isinstance(d, dict) else {}
    if seg is None:
        try:
            seg = np.load(seg_path, allow_pickle=True).item()
        except Exception:  # noqa: BLE001
            return {}
    return {k: seg[k] for k in PRED_KEYS if k in seg}


def update_pred(seg_path, **new_keys):
    """Merge prediction key/values into the sidecar (creating it if absent).

    The dataset `_seg.npy` is **never** touched — that's the whole point.
    Write is atomic-ish (tmp file + os.replace) so a crash mid-save can't
    leave a half-written sidecar.
    """
    pp = pred_path(seg_path)
    cur = {}
    if os.path.exists(pp):
        try:
            cur = np.load(pp, allow_pickle=True).item() or {}
        except Exception:  # noqa: BLE001
            cur = {}
    cur.update({k: v for k, v in new_keys.items() if v is not None})
    tmp = pp + ".tmp"
    # write via fd so np.save doesn't auto-append '.npy' to the tmp name.
    with open(tmp, "wb") as fh:
        np.save(fh, cur, allow_pickle=True)
    os.replace(tmp, pp)
    return pp


# ── image loading ──────────────────────────────────────────────────────────────

def load_image_plane(seg, tif_path, channel=1):
    """Return the 2-D MYO7A plane as float32.

    Prefers the paired TIF (more reliable than the embedded 'img', mirroring
    augment.py). label_xfer.py writes single-channel MYO7A TIFs; if a
    multi-channel stack is found the requested `channel` (default 1 = MYO7A)
    is extracted, auto-detecting (C,H,W) vs (H,W,C). Falls back to the
    brightest channel if the requested one is blank (image 011 quirk).
    """
    img = tifffile.imread(tif_path) if (tif_path and os.path.exists(tif_path)) else None
    if img is None:
        img = seg.get("img")
    if img is None:
        raise ValueError("no image data (no paired TIF and no 'img' in seg)")

    img = np.asarray(img)
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.ndim == 3:
        if img.shape[2] <= 4 and img.shape[0] > 4:  # (H,W,C) → (C,H,W)
            img = np.moveaxis(img, -1, 0)
        plane = img[channel] if channel < img.shape[0] else img[0]
        if plane.max() == 0:
            sums = img.reshape(img.shape[0], -1).sum(axis=1)
            plane = img[int(np.argmax(sums))]
        return plane.astype(np.float32)
    raise ValueError(f"unexpected image shape {img.shape}")


# ── XML fallback (only used when class_map is missing) ──────────────────────────

def parse_voc_xml(xml_path):
    """Return [(class_name, [xmin, ymin, xmax, ymax]), …] from a VOC XML.

    Degenerate boxes < 5 px on a side are skipped, matching label_xfer.py so
    the reconstructed correspondence lines up with the masks it produced.
    """
    root = ET.parse(xml_path).getroot()
    out = []
    for obj in root.findall("object"):
        name_el, bnd = obj.find("name"), obj.find("bndbox")
        if name_el is None or bnd is None:
            continue
        try:
            box = [int(bnd.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
        except (AttributeError, TypeError, ValueError):
            continue
        if box[2] - box[0] < 5 or box[3] - box[1] < 5:
            continue
        out.append(((name_el.text or "").strip(), box))
    return out


def find_xml(seg_path, data_dir, xml_dir=None):
    """Locate the original VOC XML for a seg file, or None.

    Tries <xml_dir>, the data_dir, and the seg's 'filename' directory for
    '<base>.xml' where <base> drops '_myo7a' and any aug tag.
    """
    base = base_stem(seg_stem(seg_path))
    cands = []
    for d in (xml_dir, data_dir, os.path.dirname(seg_path)):
        if d:
            cands.append(os.path.join(d, f"{base}.xml"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def derive_class_map_from_xml(masks, xml_path):
    """Assign each mask id the class of the VOC box it overlaps most.

    Reconstructs label_xfer's box→mask correspondence by maximum pixel
    overlap inside each box — order-independent and robust to dropped boxes.
    """
    class_map = {}
    for name, (x0, y0, x1, y1) in parse_voc_xml(xml_path):
        if name not in CLASS_TO_IDX:
            continue
        sub = masks[max(y0, 0):y1, max(x0, 0):x1]
        ids = sub[sub > 0]
        if ids.size == 0:
            continue
        vals, counts = np.unique(ids, return_counts=True)
        class_map[int(vals[np.argmax(counts)])] = name
    return class_map


def resolve_class_map(seg, seg_path, data_dir, xml_dir=None):
    """class_map from the seg dict, else derived from the original XML."""
    cm = seg.get("class_map")
    if isinstance(cm, dict) and cm:
        return {int(k): v for k, v in cm.items()}, "seg"
    xml_path = find_xml(seg_path, data_dir, xml_dir)
    if xml_path is None:
        return {}, "none"
    return derive_class_map_from_xml(seg["masks"], xml_path), "xml"


# ── crop expansion (the fiddly part) ───────────────────────────────────────────

def cell_bbox(masks, cell_id):
    """Half-open bbox (r0, c0, r1, c1) of `masks == cell_id`, or None if absent."""
    ys, xs = np.where(masks == cell_id)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def expand_bbox(bbox, *, pad_frac=0.5, pad_px=None):
    """Grow a bbox to add surrounding context.

    pad_px overrides pad_frac when given; otherwise pad = pad_frac * max(h, w)
    pixels are added on every side. The result is intentionally *unclamped* —
    it may fall outside the image; crop_pad() handles that.
    """
    r0, c0, r1, c1 = bbox
    h, w = r1 - r0, c1 - c0
    pad = pad_px if pad_px is not None else int(round(pad_frac * max(h, w)))
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def crop_pad(plane, r0, c0, r1, c1, fill=0.0):
    """Crop plane[r0:r1, c0:c1]; out-of-bounds pixels set to `fill`.

    Handles every edge case: the window may be partially or wholly outside
    the image and r0/c0 may be negative. Always returns exactly
    (r1-r0, c1-c0) float32.
    """
    H, W = plane.shape
    out = np.full((r1 - r0, c1 - c0), fill, dtype=np.float32)
    sr0, sc0 = max(r0, 0), max(c0, 0)
    sr1, sc1 = min(r1, H), min(c1, W)
    if sr1 > sr0 and sc1 > sc0:  # any overlap with the image at all
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = plane[sr0:sr1, sc0:sc1]
    return out


def extract_cell_crop(plane, masks, cell_id, *, out_size=64, pad_frac=0.5,
                      pad_px=None, pad_value=0.0, soft_mask=True):
    """Crop one instance into a fixed (2, out_size, out_size) float32 array.

    Channel 0 — MYO7A intensities (raw; the classifier z-scores them).
    Channel 1 — binary mask of *this* instance only, so the network knows
                which cell in the context-padded crop it must classify.

    Returns None if `cell_id` is absent from `masks`.
    """
    bbox = cell_bbox(masks, cell_id)
    if bbox is None:
        return None
    r0, c0, r1, c1 = expand_bbox(bbox, pad_frac=pad_frac, pad_px=pad_px)

    img_crop = crop_pad(plane, r0, c0, r1, c1, fill=pad_value)
    tgt = (masks == cell_id).astype(np.float32)
    msk_crop = crop_pad(tgt, r0, c0, r1, c1, fill=0.0)

    dst = (out_size, out_size)  # cv2 takes (W, H); square so it's symmetric
    img_interp = cv2.INTER_AREA if img_crop.shape[0] >= out_size else cv2.INTER_LINEAR
    msk_interp = cv2.INTER_LINEAR if soft_mask else cv2.INTER_NEAREST
    img_rs = cv2.resize(img_crop, dst, interpolation=img_interp)
    msk_rs = cv2.resize(msk_crop, dst, interpolation=msk_interp)
    if not soft_mask:
        msk_rs = (msk_rs > 0.5).astype(np.float32)

    return np.stack([img_rs, msk_rs], axis=0).astype(np.float32)


# ── dataset builder ────────────────────────────────────────────────────────────

def iter_seg_files(data_dir, include_augmented=False):
    """Yield (seg_path, group_key) for every usable _seg.npy in data_dir."""
    for seg_path in sorted(glob.glob(os.path.join(data_dir, "*_seg.npy"))):
        if not include_augmented and is_augmented(seg_stem(seg_path)):
            continue
        yield seg_path, group_key(seg_path)


def build_crop_dataset(data_dir, *, out_size=64, pad_frac=0.5, pad_px=None,
                       pad_value="mean", soft_mask=True, channel=1,
                       include_augmented=False, xml_dir=None):
    """Walk data_dir → (crops, labels, groups, image_names).

    crops  : (N, 2, S, S) float32   raw intensities
    labels : (N,)        int64       0=IHC, 1=OHC
    groups : (N,)        int64       source-image index (leak-free splits)
    image_names : list[str]          one stem per group index
    """
    crops, labels, groups, image_names = [], [], [], []
    gkey_to_idx = {}
    n_files = n_cells = n_skipped = 0
    src_counter = {"seg": 0, "xml": 0, "none": 0}

    for seg_path, gkey in iter_seg_files(data_dir, include_augmented):
        try:
            seg = np.load(seg_path, allow_pickle=True).item()
        except Exception as e:  # noqa: BLE001 — keep the batch going
            print(f"  ERROR loading {os.path.basename(seg_path)}: {e}")
            continue
        masks = seg.get("masks")
        if masks is None:
            print(f"  skip {os.path.basename(seg_path)} (no masks)")
            continue

        class_map, src = resolve_class_map(seg, seg_path, data_dir, xml_dir)
        src_counter[src] = src_counter.get(src, 0) + 1
        if not class_map:
            print(f"  skip {os.path.basename(seg_path)} (no class labels)")
            continue

        try:
            plane = load_image_plane(seg, tif_for_seg(seg_path), channel)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR image for {os.path.basename(seg_path)}: {e}")
            continue

        fill = float(plane.mean()) if pad_value == "mean" else float(pad_value)
        if gkey not in gkey_to_idx:
            gkey_to_idx[gkey] = len(image_names)
            image_names.append(seg_stem(seg_path))
        gidx = gkey_to_idx[gkey]

        kept = 0
        for cell_id, cls in class_map.items():
            if cls not in CLASS_TO_IDX:
                n_skipped += 1
                continue
            crop = extract_cell_crop(
                plane, masks, int(cell_id), out_size=out_size,
                pad_frac=pad_frac, pad_px=pad_px, pad_value=fill,
                soft_mask=soft_mask,
            )
            if crop is None:  # mask erased by overlap resolution upstream
                n_skipped += 1
                continue
            crops.append(crop)
            labels.append(CLASS_TO_IDX[cls])
            groups.append(gidx)
            kept += 1
        n_files += 1
        n_cells += kept
        print(f"  {seg_stem(seg_path):52s} {kept:4d} cells  grp{gidx:>3}  [{src}]")

    if not crops:
        raise RuntimeError(f"no crops produced from {data_dir}")

    crops = np.stack(crops).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    n_ihc, n_ohc = int((labels == 0).sum()), int((labels == 1).sum())
    print(f"\n{n_files} files, {len(image_names)} source images, {n_cells} crops "
          f"(IHC {n_ihc} / OHC {n_ohc}); {n_skipped} skipped")
    print(f"label source: {src_counter}")
    return crops, labels, groups, image_names


def save_dataset(path, crops, labels, groups, image_names, meta):
    """Write a compressed .npz the classifier loads directly."""
    np.savez_compressed(
        path,
        crops=crops, labels=labels, groups=groups,
        image_names=np.asarray(image_names, dtype=object),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
        meta=np.asarray([meta], dtype=object),
    )
    print(f"saved → {path}  {crops.shape}  ({os.path.getsize(path) / 1e6:.1f} MB)")


# ── sanity-check montage ───────────────────────────────────────────────────────

def save_preview(path, crops, labels, per_class=24, seed=0):
    """Montage of random crops per class — eyeball before training."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("preview skipped (matplotlib not installed)")
        return

    rng = np.random.default_rng(seed)
    cols = 8
    rows_pc = (per_class + cols - 1) // cols
    fig, axes = plt.subplots(
        len(CLASS_NAMES) * rows_pc, cols * 2,
        figsize=(cols * 2 * 1.3, len(CLASS_NAMES) * rows_pc * 1.3),
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.axis("off")

    for ci, cls in enumerate(CLASS_NAMES):
        idx = np.where(labels == ci)[0]
        if idx.size == 0:
            continue
        pick = rng.choice(idx, size=min(per_class, idx.size), replace=False)
        for j, k in enumerate(pick):
            r, c = divmod(j, cols)
            img, msk = crops[k]
            lo, hi = np.percentile(img, (1, 99))
            disp = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
            ar = ci * rows_pc + r
            axes[ar][c * 2].imshow(disp, cmap="gray")
            axes[ar][c * 2].set_title(cls, fontsize=6)
            axes[ar][c * 2 + 1].imshow(msk, cmap="magma", vmin=0, vmax=1)
            axes[ar][c * 2 + 1].set_title("mask", fontsize=6)

    fig.suptitle("IHC/OHC crops — image | target-mask channel", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"preview → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build IHC/OHC crop dataset from _seg.npy files")
    p.add_argument("--data_dir", required=True,
                   help="Directory of <stem>_myo7a_seg.npy (+ .tif) files")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--out_size", type=int, default=64, help="Square crop size (default 64)")
    p.add_argument("--pad_frac", type=float, default=0.5,
                   help="Context padding as a fraction of max(cell h, w) (default 0.5)")
    p.add_argument("--pad_px", type=int, default=None,
                   help="Fixed padding in px; overrides --pad_frac when set")
    p.add_argument("--pad_value", default="mean",
                   help="'mean' (per-image) or a float for out-of-bounds fill")
    p.add_argument("--hard_mask", action="store_true",
                   help="Binarise the resized mask channel (default: soft)")
    p.add_argument("--channel", type=int, default=1,
                   help="Channel index if a multi-channel TIF is found (default 1=MYO7A)")
    p.add_argument("--include-augmented", action="store_true",
                   help="Also ingest augment.py's on-disk D4 copies")
    p.add_argument("--xml_dir", default=None,
                   help="Directory of original VOC XMLs (fallback when class_map absent)")
    p.add_argument("--preview", default=None, metavar="PNG",
                   help="Also write a crop montage to this PNG for sanity checks")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Building crops from {args.data_dir}")
    crops, labels, groups, image_names = build_crop_dataset(
        args.data_dir, out_size=args.out_size, pad_frac=args.pad_frac,
        pad_px=args.pad_px, pad_value=args.pad_value,
        soft_mask=not args.hard_mask, channel=args.channel,
        include_augmented=args.include_augmented, xml_dir=args.xml_dir,
    )
    meta = dict(
        out_size=args.out_size, pad_frac=args.pad_frac, pad_px=args.pad_px,
        pad_value=args.pad_value, soft_mask=not args.hard_mask,
        channel=args.channel, data_dir=os.path.abspath(args.data_dir),
    )
    save_dataset(args.out, crops, labels, groups, image_names, meta)
    if args.preview:
        save_preview(args.preview, crops, labels)


if __name__ == "__main__":
    main()
