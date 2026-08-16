"""Partition the RESIDUAL false negatives: merged, or missed outright?

The 'residual ~28% FN is merging of densely-packed cells' claim in progress.md
rests on the morphology of missed cells, the threshold sweep, and figures — never
on a direct measurement. This is that measurement.

FN set = GT cells the curator hand-drew (`ismanual`), i.e. cells stock cpsam
missed. Residual FN = those the fine-tuned model still fails to recover
(best IoU vs any prediction < 0.5). For each, ask where its centroid lands:

  MERGED          inside a predicted mask -> the model sees tissue there but
                  lumps it with a neighbour. A split-proposer is well-posed.
  MISSED OUTRIGHT in background -> nothing predicted at all. A mask-level
                  checker/splitter is structurally unable to help; the problem
                  is detection.

Each image is scored by the fold model that HELD IT OUT (manifest v4).
"""
import argparse, glob, json, os
import numpy as np
import tifffile
from cellpose import models as cpmodels
from scipy.ndimage import center_of_mass

AUG = ("_SV_", "_rot90", "_rot180", "_rot270", "_fliph", "_flipv")


def stem(p):
    return os.path.basename(p)[:-len("_seg.npy")]


def fold_of(manifest, s):
    for f in manifest["folds"]:
        if s in f["test"]:
            return f["fold"]
    return None


def best_iou(gt_bin, pred, cand):
    best = 0.0
    for l in cand:
        pm = pred == l
        inter = np.logical_and(gt_bin, pm).sum()
        if inter:
            best = max(best, inter / np.logical_or(gt_bin, pm).sum())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model_pattern", required=True, help="path with {k}")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()
    man = json.load(open(a.manifest))

    segs = [p for d in a.data_dirs.split(",")
            for p in sorted(glob.glob(os.path.join(d.strip(), "*_seg.npy")))]
    segs = [p for p in segs if not any(m in stem(p) for m in AUG)]
    by_fold = {}
    for p in segs:
        k = fold_of(man, stem(p))
        if k is not None:
            by_fold.setdefault(k, []).append(p)

    tot = {"merged": 0, "missed": 0, "recovered": 0}
    per_img = []
    for k in sorted(by_fold):
        model = cpmodels.CellposeModel(gpu=True, pretrained_model=a.model_pattern.format(k=k))
        for p in by_fold[k]:
            d = np.load(p, allow_pickle=True).item()
            gt, ism = d["masks"], np.asarray(d["ismanual"]).ravel()
            img = tifffile.imread(os.path.join(os.path.dirname(p), stem(p) + ".tif"))
            ca = 2 if img.ndim == 3 else None
            pred = model.eval(img, channel_axis=ca, normalize=True)[0].astype(np.int32)
            labels = np.unique(gt)[1:]
            fn_labels = labels[ism[labels - 1]] if ism.size >= labels.max() else labels
            m = {"merged": 0, "missed": 0, "recovered": 0}
            for l in fn_labels:
                gb = gt == l
                cand = np.unique(pred[gb])
                cand = cand[cand > 0]
                if len(cand) and best_iou(gb, pred, cand) >= a.iou:
                    m["recovered"] += 1
                    continue
                cy, cx = center_of_mass(gb)
                inside = pred[int(round(cy)), int(round(cx))] > 0
                m["merged" if inside else "missed"] += 1
            for key in tot:
                tot[key] += m[key]
            per_img.append((stem(p), len(fn_labels), m))
            print(f"  fold{k} {stem(p)[:44]:<44} FN={len(fn_labels):3d} "
                  f"recovered={m['recovered']:3d} merged={m['merged']:3d} missed={m['missed']:3d}",
                  flush=True)

    res = tot["merged"] + tot["missed"]
    print("\n================ RESIDUAL FN PARTITION ================")
    print(f"hand-drawn FN cells total : {sum(tot.values())}")
    print(f"  recovered by the model  : {tot['recovered']}  ({tot['recovered']/max(sum(tot.values()),1):.1%})")
    print(f"  RESIDUAL FN             : {res}")
    if res:
        print(f"     MERGED (centroid inside a predicted mask) : {tot['merged']:5d}  {tot['merged']/res:.1%}")
        print(f"     MISSED OUTRIGHT (centroid in background)  : {tot['missed']:5d}  {tot['missed']/res:.1%}")
        print("\n=> " + ("merging dominates: a split-proposer is well-posed"
                         if tot["merged"] > tot["missed"] else
                         "outright misses dominate: mask-level splitting CANNOT help; "
                         "the gap is detection-side"))


if __name__ == "__main__":
    main()
