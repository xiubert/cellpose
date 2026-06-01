"""
Head-to-head segmentation evaluation for Cellpose-SAM models.

Runs one or more models on a test directory of paired <stem>.tif + <stem>_seg.npy
(the _seg.npy supplies the ground-truth instance masks), and reports
**average precision** at IoU thresholds [0.5, 0.75, 0.9] plus mean true/pred
cell counts — the standard instance-segmentation metric
(cellpose.metrics.average_precision).

This is the ranking metric for the CLC experiment plan
(helpers/notes/clc_seg_experiments.md): pick models by test AP@0.5, watch
count drift (mean n_pred vs n_true) as the over/under-segmentation tell.

Usage:
  python eval_seg.py --test_dir DIR --models name1,name2 \
      [--models_dir /ix1/pcody/cellpose/models] \
      [--thresholds 0.5,0.75,0.9] [--out results.csv] [--cpu]

Each --models entry is used as a path if it exists, else joined with
--models_dir. Augmented copies (_SV / _rot / _flip stems) are skipped so the
test set is originals only.
"""

import argparse
import glob
import os

import numpy as np
import tifffile
from cellpose import models
from cellpose.metrics import average_precision

_AUG_MARKERS = ("_SV_", "_rot90", "_rot180", "_rot270", "_fliph", "_flipv")


def _seg_stem(p):
    b = os.path.basename(p)
    return b[:-len("_seg.npy")] if b.endswith("_seg.npy") else os.path.splitext(b)[0]


def _is_aug(stem):
    return any(m in stem for m in _AUG_MARKERS)


def iter_test_segs(test_dir):
    segs = sorted(glob.glob(os.path.join(test_dir, "*_seg.npy")))
    return [p for p in segs if not _is_aug(_seg_stem(p))]


def load_pair(seg_path):
    """Return (image HxWx[C], gt_masks HxW int) for a test seg file."""
    d = np.load(seg_path, allow_pickle=True).item()
    gt = d["masks"]
    tif = os.path.join(os.path.dirname(seg_path), _seg_stem(seg_path) + ".tif")
    img = tifffile.imread(tif) if os.path.exists(tif) else d.get("img")
    if img is None:
        raise ValueError(f"no image for {seg_path}")
    return img, gt


def resolve_model(entry, models_dir):
    if os.path.exists(entry):
        return entry
    cand = os.path.join(models_dir, entry)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"model not found: {entry} (also tried {cand})")


def parse_args():
    p = argparse.ArgumentParser(description="Head-to-head Cellpose-SAM segmentation eval (AP@IoU)")
    p.add_argument("--test_dir", required=True)
    p.add_argument("--models", required=True, help="comma-separated model names/paths")
    p.add_argument("--models_dir", default="/ix1/pcody/cellpose/models")
    p.add_argument("--thresholds", default="0.5,0.75,0.9")
    p.add_argument("--out", default=None, help="CSV path for per-image results")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cpu", action="store_true", help="force CPU (default: GPU)")
    return p.parse_args()


def main():
    args = parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]
    model_entries = [m.strip() for m in args.models.split(",") if m.strip()]

    seg_paths = iter_test_segs(args.test_dir)
    if not seg_paths:
        raise SystemExit(f"no (non-augmented) *_seg.npy in {args.test_dir}")
    print(f"test images: {len(seg_paths)}  thresholds: {thresholds}\n", flush=True)

    images, gts, names = [], [], []
    for sp in seg_paths:
        img, gt = load_pair(sp)
        images.append(img)
        gts.append(gt)
        names.append(_seg_stem(sp))
    n_true = np.array([len(np.unique(g)) - 1 for g in gts])

    channel_axis = 2 if images[0].ndim == 3 else None
    rows = []          # (model, image, n_true, n_pred, ap@th...)
    summary = []       # (model, mAP@th..., mean_n_pred)

    for entry in model_entries:
        path = resolve_model(entry, args.models_dir)
        name = os.path.basename(path)
        print(f"=== {name} ===", flush=True)
        model = models.CellposeModel(gpu=not args.cpu, pretrained_model=path)
        masks_pred, _, _ = model.eval(images, batch_size=args.batch_size,
                                      channel_axis=channel_axis, normalize=True)
        ap, tp, fp, fn = average_precision(gts, masks_pred, threshold=thresholds)
        n_pred = np.array([len(np.unique(m)) - 1 for m in masks_pred])
        for i, nm in enumerate(names):
            rows.append([name, nm, int(n_true[i]), int(n_pred[i])] + [float(ap[i, k]) for k in range(len(thresholds))])
        mAP = ap.mean(axis=0)
        summary.append([name] + [float(v) for v in mAP] + [float(n_pred.mean())])
        thstr = "  ".join(f"AP@{t}={mAP[k]:.4f}" for k, t in enumerate(thresholds))
        print(f"  {thstr}   mean cells true={n_true.mean():.1f} pred={n_pred.mean():.1f}\n", flush=True)

    # --- comparison table ---
    th_cols = "  ".join(f"AP@{t}" for t in thresholds)
    print("=" * 72)
    print(f"{'model':<32} {th_cols}   n_pred (true={n_true.mean():.1f})")
    print("-" * 72)
    for s in summary:
        aps = "  ".join(f"{v:.4f}" for v in s[1:1 + len(thresholds)])
        print(f"{s[0]:<32} {aps}   {s[-1]:.1f}")
    print("=" * 72)

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "image", "n_true", "n_pred"] + [f"AP@{t}" for t in thresholds])
            w.writerows(rows)
        print(f"\nwrote per-image results: {args.out}")


if __name__ == "__main__":
    main()
