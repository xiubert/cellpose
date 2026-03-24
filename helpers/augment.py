"""
Augment cellpose training data (TIF + _seg.npy pairs).

Applies the 8 isometries of the square (D4: rotations × flips) plus optional
intensity jitter to every <stem>_seg.npy / <stem>.tif pair found in data_dir.

Outputs per source image:
  <stem>_rot90.<ext>         k=1 counter-clockwise rotation
  <stem>_rot180.<ext>
  <stem>_rot270.<ext>
  <stem>_fliph.<ext>         horizontal flip (left ↔ right)
  <stem>_flipv.<ext>         vertical flip (top ↔ bottom)
  <stem>_rot90_fliph.<ext>
  <stem>_rot90_flipv.<ext>

The original is not re-saved.  Masks use integer-preserving numpy ops (no
interpolation), so label values are exactly preserved.  Outlines are
recomputed from each transformed mask.

Usage:
  python augment.py --data_dir /path/to/data [--out_dir /path/to/output]
                    [--suffix ch1]           # only process files ending in ch1
                    [--intensity]            # also vary brightness/gamma
                    [--seed 42]
"""

import argparse
import glob
import os

import numpy as np
import tifffile
from cellpose.utils import masks_to_outlines


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="D4 augmentation for cellpose training data")
    p.add_argument("--data_dir", required=True,
                   help="Directory containing *_seg.npy (and matching *.tif) files")
    p.add_argument("--out_dir", default=None,
                   help="Output directory; defaults to data_dir")
    p.add_argument("--suffix", default=None,
                   help="Only process seg files whose stem ends with this suffix, "
                        "e.g. 'ch1' matches foo_ch1_seg.npy")
    p.add_argument("--intensity", action="store_true",
                   help="Apply random brightness/gamma jitter in addition to spatial transforms")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for intensity jitter")
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = args.data_dir
    return args


# ── spatial transforms (lossless, integer-safe) ─────────────────────────────────

def _rot90(arr, k):
    """np.rot90 across the last two axes (works for 2-D and for future 3-D stacks)."""
    return np.rot90(arr, k=k, axes=(-2, -1))


def _fliph(arr):
    """Flip left ↔ right (last axis)."""
    return np.flip(arr, axis=-1)


def _flipv(arr):
    """Flip top ↔ bottom (second-to-last axis)."""
    return np.flip(arr, axis=-2)


# Named transform list: (tag, fn)
SPATIAL_TRANSFORMS = [
    ("rot90",       lambda a: _rot90(a, 1)),
    ("rot180",      lambda a: _rot90(a, 2)),
    ("rot270",      lambda a: _rot90(a, 3)),
    ("fliph",       _fliph),
    ("flipv",       _flipv),
    ("rot90_fliph", lambda a: _fliph(_rot90(a, 1))),
    ("rot90_flipv", lambda a: _flipv(_rot90(a, 1))),
]


# ── intensity augmentation ────────────────────────────────────────────────────────

def apply_intensity_jitter(img_u16, rng):
    """
    Random brightness scale + gamma on a uint16 image.
    Returns a new uint16 array; does NOT clip to original max.
    """
    img = img_u16.astype(np.float32)
    # brightness: multiply by U[0.75, 1.25]
    scale = rng.uniform(0.75, 1.25)
    img = img * scale
    # gamma: raise to U[0.75, 1.33]
    gamma = rng.uniform(0.75, 1.33)
    hi = float(img.max()) or 1.0
    img = (img / hi) ** gamma * hi
    return np.clip(img, 0, 65535).astype(np.uint16)


# ── per-pair processing ──────────────────────────────────────────────────────────

def augment_pair(seg_path, tif_path, out_dir, do_intensity, rng):
    seg_data = np.load(seg_path, allow_pickle=True).item()
    masks_orig = seg_data["masks"]       # (H, W) int32
    img_orig   = seg_data.get("img")     # (H, W) uint16 or None
    class_map  = seg_data.get("class_map", {})

    # Prefer loading from the paired TIF (more reliable than the embedded img)
    if tif_path and os.path.exists(tif_path):
        img_orig = tifffile.imread(tif_path)

    if img_orig is None:
        raise ValueError(f"No image data for {seg_path}")

    stem = _seg_stem(seg_path)           # e.g.  "foo_ch1"

    n_saved = 0
    for tag, tfm in SPATIAL_TRANSFORMS:
        new_stem    = f"{stem}_{tag}"
        out_tif_path = os.path.join(out_dir, f"{new_stem}.tif")
        out_seg_path = os.path.join(out_dir, f"{new_stem}_seg.npy")

        if os.path.exists(out_seg_path):
            continue  # skip already-done

        # --- spatial transform ---
        masks_aug = np.ascontiguousarray(tfm(masks_orig))
        img_aug   = np.ascontiguousarray(tfm(img_orig))

        # --- optional intensity jitter ---
        if do_intensity:
            img_aug = apply_intensity_jitter(img_aug, rng)

        # --- recompute outlines from transformed mask ---
        outlines_aug = masks_to_outlines(masks_aug)

        # --- save TIF ---
        tifffile.imwrite(out_tif_path, img_aug)

        # --- save seg ---
        np.save(
            out_seg_path,
            {
                "masks":     masks_aug,
                "outlines":  outlines_aug,
                "filename":  out_tif_path,
                "img":       img_aug,
                "class_map": class_map,
            },
        )
        n_saved += 1

    return n_saved


def _seg_stem(seg_path):
    """'foo/bar_ch1_seg.npy'  →  'bar_ch1'"""
    base = os.path.basename(seg_path)       # bar_ch1_seg.npy
    if base.endswith("_seg.npy"):
        return base[: -len("_seg.npy")]
    return os.path.splitext(base)[0]


def _tif_for_seg(seg_path):
    """Try <stem>.tif next to the seg file."""
    stem = _seg_stem(seg_path)
    folder = os.path.dirname(seg_path)
    return os.path.join(folder, f"{stem}.tif")


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    rng   = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    seg_paths = sorted(glob.glob(os.path.join(args.data_dir, "*_seg.npy")))

    # Filter to augmentation-tag-free originals only (skip already-augmented files)
    aug_tags = {t for t, _ in SPATIAL_TRANSFORMS}
    seg_paths = [
        p for p in seg_paths
        if not any(_seg_stem(p).endswith(f"_{tag}") for tag in aug_tags)
    ]

    if args.suffix:
        seg_paths = [p for p in seg_paths if _seg_stem(p).endswith(args.suffix)]

    print(f"Found {len(seg_paths)} source seg files")

    total_saved = 0
    for seg_path in seg_paths:
        tif_path = _tif_for_seg(seg_path)
        stem     = _seg_stem(seg_path)
        print(f"Augmenting {stem} …", end=" ", flush=True)
        try:
            n = augment_pair(seg_path, tif_path, args.out_dir, args.intensity, rng)
            print(f"{n} new pairs")
            total_saved += n
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone — {total_saved} augmented pairs written to {args.out_dir}")


if __name__ == "__main__":
    main()
