"""
Validate the hair-cell mask post-process (`density_outlier_reject`) — the
spatial-outlier reject that deletes masks sitting far from the main
organ-of-Corti rows of cells. Mirrors `clc_error_eval.py`'s discipline:
we want an operating point that catches false-positive masks while deleting
**zero** real cells, exactly as `row_consistency` was locked on the curator
corrections.

Two modes:

  --mode gt   (no GPU, no model — run this first)
      Run the reject pass on the CURATED GT segs. Every mask in a GT seg is
      a real, curator-kept or curator-added cell, so any mask the pass flags
      is a FALSE DELETION. This is the TP-loss column, measured directly and
      leak-free (the reject is unsupervised geometry, never saw labels). The
      hand-added cells (ismanual=True — the hard, sometimes-isolated large
      OHCs) are the ones most at risk, so they're reported separately as a
      stress test.

  --mode model   (GPU; needs --model)
      Run a segmentation model on each tif → predicted masks (which contain
      FPs) → label each pred mask TP (IoU>=thr to a GT cell) or FP → run the
      reject pass. Reports FP-catch (good) and TP-loss (bad) on the model's
      own predictions. Use a held-out fold model per image for a leak-free
      FP-catch estimate; the all-data deploy model is in-sample (its FP set
      is smaller) so treat its FP-catch as a lower bound.

Usage (container):
  podman exec cellpose python3 /helpers/clc_reject_eval.py --mode gt \
      --data_dirs /data/cellpose_cc/adult,/data/cellpose_cc/neonate
  podman exec cellpose python3 /helpers/clc_reject_eval.py --mode model \
      --model /data/clc_all_full_deploy \
      --data_dirs /data/cellpose_cc/adult,/data/cellpose_cc/neonate
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ihc_ohc"))
import clc_split  # noqa: E402  seg_stem / is_aug
from ihc_ohc_geom import reject_for_seg  # noqa: E402


# Operating points to sweep (label → reject_for_seg kwargs).
# gap = min distance (in D) an off-band cluster must sit from the band; frac =
# max satellite size as a fraction of the band (protects a real second segment).
OPS = [
    ("gap3 frac0.5",  dict(eps=2.5, min_gap=3.0, max_frac=0.5)),
    ("gap4 frac0.5",  dict(eps=2.5, min_gap=4.0, max_frac=0.5)),
    ("gap5 frac0.5",  dict(eps=2.5, min_gap=5.0, max_frac=0.5)),
    ("gap6 frac0.4",  dict(eps=2.5, min_gap=6.0, max_frac=0.4)),
    ("gap4 frac0.3",  dict(eps=2.5, min_gap=4.0, max_frac=0.3)),
    ("gap8 frac0.3",  dict(eps=2.5, min_gap=8.0, max_frac=0.3)),
]


def best_iou_to_gt(pred_bin, gt):
    """IoU of a predicted mask to its best-overlapping GT mask (0 if none)."""
    ov = gt[pred_bin]
    ov = ov[ov > 0]
    if ov.size == 0:
        return 0.0
    gm = gt == np.bincount(ov).argmax()
    return (pred_bin & gm).sum() / (pred_bin | gm).sum()


def collect_segs(data_dirs):
    segs = []
    for d in data_dirs.split(","):
        for p in sorted(glob.glob(os.path.join(d.strip(), "*_seg.npy"))):
            if not clc_split.is_aug(clc_split.seg_stem(p)):   # originals only
                segs.append(p)
    return segs


# ── GT mode: flags on curated GT = false deletions ──────────────────────────────

def run_gt(segs):
    print(f"[gt] {len(segs)} curated segs — every flagged mask is a FALSE "
          f"deletion of a real cell\n")
    # accumulators per op
    tot_cells = 0
    tot_added = 0
    flagged = {name: 0 for name, _ in OPS}
    flagged_added = {name: 0 for name, _ in OPS}
    worst = {name: [] for name, _ in OPS}   # (n_flagged, image) for context

    for sp in segs:
        d = np.load(sp, allow_pickle=True).item()
        masks = d.get("masks")
        if masks is None or masks.max() == 0:
            continue
        ism = np.asarray(d.get("ismanual")) if d.get("ismanual") is not None \
            else np.zeros(int(masks.max()), bool)
        n = int(masks.max())
        tot_cells += n
        tot_added += int(ism.sum())
        for name, kw in OPS:
            rej = reject_for_seg({"masks": masks}, **kw)
            if rej:
                flagged[name] += len(rej)
                # ismanual indexed by cid-1
                fa = sum(1 for c in rej if 0 <= c - 1 < len(ism) and ism[c - 1])
                flagged_added[name] += fa
                worst[name].append((len(rej), os.path.basename(sp)[:38]))

    print(f"total real cells: {tot_cells}   (hand-added: {tot_added})\n")
    print(f"{'operating point':<24} {'false-del':>9} {'rate':>7} "
          f"{'of-added':>9}")
    print("-" * 54)
    for name, _ in OPS:
        fd = flagged[name]
        print(f"{name:<24} {fd:>9} {fd / max(tot_cells, 1):>6.2%} "
              f"{flagged_added[name]:>9}")
    print("\nper-image false deletions (top offenders):")
    for name, _ in OPS:
        w = sorted(worst[name], reverse=True)[:3]
        tag = ", ".join(f"{n}×[{im}]" for n, im in w) or "none"
        print(f"  {name:<24} {tag}")


# ── model mode: FP-catch vs TP-loss on model predictions ────────────────────────

def run_model_mode(segs, model_path, iou_thr):
    import tifffile
    from cellpose import models as cpmodels
    m = cpmodels.CellposeModel(gpu=True, pretrained_model=model_path)
    print(f"[model] {os.path.basename(model_path)}  IoU thr {iou_thr}  "
          f"{len(segs)} images\n")

    tot_tp = tot_fp = 0
    catch = {name: 0 for name, _ in OPS}     # FP preds flagged (good)
    loss = {name: 0 for name, _ in OPS}      # TP preds flagged (bad)

    for sp in segs:
        d = np.load(sp, allow_pickle=True).item()
        gt = d.get("masks")
        if gt is None or gt.max() == 0:
            continue
        stem = clc_split.seg_stem(sp)
        tif = os.path.join(os.path.dirname(sp), stem + ".tif")
        img = tifffile.imread(tif) if os.path.exists(tif) else d.get("img")
        if img is None:
            print(f"  skip (no image): {os.path.basename(sp)[:40]}")
            continue
        ca = 2 if img.ndim == 3 else None
        pred = m.eval(img, channel_axis=ca, normalize=True)[0].astype(np.int32)
        pids = np.unique(pred)
        pids = pids[pids > 0]
        # label each predicted mask TP / FP by IoU to GT
        is_fp = {int(pid): best_iou_to_gt(pred == pid, gt) < iou_thr
                 for pid in pids}
        n_fp = sum(is_fp.values())
        n_tp = len(pids) - n_fp
        tot_fp += n_fp
        tot_tp += n_tp
        for name, kw in OPS:
            rej = reject_for_seg({"masks": pred}, **kw)
            for c in rej:
                if is_fp.get(int(c), False):
                    catch[name] += 1
                else:
                    loss[name] += 1
        print(f"  {os.path.basename(sp)[:40]:40s}  TP={n_tp:3d} FP={n_fp:3d}",
              flush=True)

    print(f"\ntotal predicted: TP={tot_tp}  FP={tot_fp}\n")
    print(f"{'operating point':<24} {'FP-catch':>17} {'TP-loss':>17}")
    print("-" * 60)
    for name, _ in OPS:
        c, l = catch[name], loss[name]
        print(f"{name:<24} {c:>6}/{tot_fp:<5} ({c/max(tot_fp,1):>5.1%}) "
              f"{l:>6}/{tot_tp:<5} ({l/max(tot_tp,1):>5.2%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gt", "model"], default="gt")
    ap.add_argument("--data_dirs", required=True,
                    help="comma-separated dirs of *_seg.npy")
    ap.add_argument("--model", default=None, help="seg model for --mode model")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    segs = collect_segs(args.data_dirs)
    if not segs:
        print("no segs found")
        return
    if args.mode == "gt":
        run_gt(segs)
    else:
        if not args.model:
            ap.error("--mode model requires --model")
        run_model_mode(segs, args.model, args.iou)


if __name__ == "__main__":
    main()
