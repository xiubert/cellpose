"""
Over-detect + filter threshold sweep (Approach B follow-on).

The curation error-eval showed fine-tuning suppresses 91% of cpsam's false
positives but still misses ~44% of the cells cpsam missed (FN). Hypothesis:
lowering the detection threshold on the FINE-TUNED model (which now has strong
learned precision) recovers those missed cells without flooding new FPs.

This sweeps `cellprob_threshold` (lower => more detections) and `flow_threshold`
(higher => more permissive) on the LOAO held-out fold models, and reports — per
grid point, aggregated over the 15 CLC images:

  FN-recov   frac of hand-added (ismanual=True, cpsam-missed) cells detected   ↑
  TP-ret     frac of kept (ismanual=False) cells detected                      ↑
  modelFP    mean # predicted masks per image NOT matching any curated GT mask ↓
             (the model's OWN false positives — what over-detection costs)
  AP@0.5     mean per-image AP vs the full curated GT (net precision/recall)   ↑

The default point (flow 0.4, cellprob 0.0) reproduces the Approach-B numbers.
The best AP@0.5 row is the operating point; if FN-recov rises but AP falls, the
new FPs need a learned filter (over-detect THEN filter).

Usage (cluster):
  python clc_threshold_sweep.py \
      --data_dirs /ix1/.../CLC/adult,/ix1/.../CLC/neonate \
      --manifest /ix1/.../CLC/clc_folds_loao.json \
      --model_pattern /ix1/pcody/cellpose/models/clc_all_warm_20260602_1353_fold{k} \
      --cellprob 0.0,-2.0,-4.0,-6.0 --flow 0.4,0.8
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import tifffile
from cellpose import models as cpmodels
from cellpose.metrics import average_precision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clc_split

AUG = clc_split.is_aug


def label_ids(mask):
    u = np.unique(mask)
    return u[u > 0]


def load(seg_path):
    d = np.load(seg_path, allow_pickle=True).item()
    masks = d["masks"].astype(np.int32)
    ism = np.asarray(d.get("ismanual"))
    tif = os.path.join(os.path.dirname(seg_path), clc_split.seg_stem(seg_path) + ".tif")
    img = tifffile.imread(tif) if os.path.exists(tif) else d.get("img")
    return img, masks, ism


def best_iou(target_bin, pred):
    ov = pred[target_bin]; ov = ov[ov > 0]
    if ov.size == 0:
        return 0.0
    pm = pred == np.bincount(ov).argmax()
    return (pm & target_bin).sum() / (pm | target_bin).sum()


def fold_for_animal(manifest, animal):
    for f in manifest["folds"]:
        if animal in f["test_animals"]:
            return f["fold"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model_pattern", required=True, help="path with {k} for held-out fold")
    ap.add_argument("--cellprob", default="0.0,-2.0,-4.0,-6.0")
    ap.add_argument("--flow", default="0.4,0.8")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    thr = args.iou
    cps = [float(x) for x in args.cellprob.split(",")]
    fls = [float(x) for x in args.flow.split(",")]

    segs = [p for d in args.data_dirs.split(",") for p in sorted(glob.glob(os.path.join(d.strip(), "*_seg.npy")))]
    segs = [p for p in segs if not AUG(clc_split.seg_stem(p))]
    print(f"images: {len(segs)}  grid: flow{fls} x cellprob{cps}  IoU {thr}\n", flush=True)

    # preload per-image data + the held-out model object (cached by path)
    model_cache = {}
    items = []
    for sp in segs:
        img, masks, ism = load(sp)
        animal = clc_split.parse_animal(clc_split.seg_stem(sp))
        labels = label_ids(masks); man = ism[labels - 1]
        k = fold_for_animal(manifest, animal)
        path = args.model_pattern.format(k=k)
        if path not in model_cache:
            model_cache[path] = cpmodels.CellposeModel(gpu=True, pretrained_model=path)
        items.append(dict(img=img, masks=masks, tp=labels[~man], fn=labels[man], model=model_cache[path]))

    # accumulate metrics per grid point
    print(f"{'flow':>5} {'cellprob':>9} {'FN-recov':>9} {'TP-ret':>7} {'modelFP':>8} {'AP@0.5':>7}")
    print("-" * 52)
    best = None
    for fl in fls:
        for cp in cps:
            rec = []; ret = []; fp_counts = []; gts = []; preds = []
            for it in items:
                ca = 2 if it["img"].ndim == 3 else None
                pred = it["model"].eval(it["img"], channel_axis=ca, normalize=True,
                                        flow_threshold=fl, cellprob_threshold=cp)[0].astype(np.int32)
                for l in it["fn"]:
                    rec.append(best_iou(it["masks"] == l, pred) >= thr)
                for l in it["tp"]:
                    ret.append(best_iou(it["masks"] == l, pred) >= thr)
                # model's own FP = predicted labels not matching any GT mask
                gt = it["masks"]; nfp = 0
                for pl in label_ids(pred):
                    pm = pred == pl
                    ov = gt[pm]; ov = ov[ov > 0]
                    iou = 0.0
                    if ov.size:
                        gm = gt == np.bincount(ov).argmax()
                        iou = (pm & gm).sum() / (pm | gm).sum()
                    if iou < thr:
                        nfp += 1
                fp_counts.append(nfp)
                gts.append(gt); preds.append(pred)
            apv = float(np.mean(average_precision(gts, preds, threshold=[thr])[0]))
            fnr = float(np.mean(rec)); tpr = float(np.mean(ret)); mfp = float(np.mean(fp_counts))
            print(f"{fl:>5.1f} {cp:>9.1f} {fnr:>8.1%} {tpr:>7.1%} {mfp:>8.1f} {apv:>7.3f}", flush=True)
            if best is None or apv > best[0]:
                best = (apv, fl, cp, fnr, tpr, mfp)
    print(f"\nBEST AP@0.5={best[0]:.3f} at flow={best[1]} cellprob={best[2]}  "
          f"(FN-recov {best[3]:.1%}, TP-ret {best[4]:.1%}, modelFP {best[5]:.1f}/img)")


if __name__ == "__main__":
    main()
