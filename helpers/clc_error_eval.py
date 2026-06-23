"""
Approach B — evaluate models against the CLC CURATION as a labeled error set.

The CLC GT was made by running cpsam in the GUI then curating: a human KEPT good
cpsam masks (ismanual=False), HAND-ADDED cells cpsam missed (ismanual=True), and
REMOVED cpsam false positives. This script measures whether a fine-tuned model
actually fixes those documented errors — more targeted than aggregate AP.

Error sets per image (clean definitions):
  TP  = final GT masks with ismanual=False   (cpsam proposed, human kept)
  FN  = final GT masks with ismanual=True     (cpsam MISSED -> human drew)
  FP  = cpsam predictions with no IoU>=thr match to any final GT mask
        (= the false positives curation removed; derived from cpsam, not the
         noisy manual_changes action log)

Metrics per evaluated model M (match = a predicted mask with IoU>=thr):
  FN-recovery   = frac of FN cells M now detects        (higher = better)
  FP-suppression= frac of cpsam-FP locations M avoids    (higher = better;
                  cpsam itself = 0 by construction = sanity check)
  TP-retention  = frac of TP cells M still detects       (should stay ~1)

Models: pass --models as comma list. Use "cpsam", a model name/path, or
"heldout:<pattern-with-{k}>" to pick the LOAO held-out fold model per image
(leak-free) via --manifest (clc_folds_loao.json). cpsam is always run (FP ref).

Usage (cluster):
  python clc_error_eval.py --data_dirs /ix1/.../CLC/adult,/ix1/.../CLC/neonate \
      --manifest /ix1/.../CLC/clc_folds_loao.json \
      --models cpsam,"heldout:/ix1/pcody/cellpose/models/clc_all_warm_20260602_1353_fold{k}"
"""

import argparse
import glob
import os
import sys

import numpy as np
import tifffile
from cellpose import models as cpmodels

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clc_split  # parse_animal, seg_stem, is_aug

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


def best_iou_to_pred(target_bin, pred):
    ov = pred[target_bin]; ov = ov[ov > 0]
    if ov.size == 0:
        return 0.0
    pm = pred == np.bincount(ov).argmax()
    return (pm & target_bin).sum() / (pm | target_bin).sum()


def best_iou_to_gt(pred_bin, gt):
    ov = gt[pred_bin]; ov = ov[ov > 0]
    if ov.size == 0:
        return 0.0
    gm = gt == np.bincount(ov).argmax()
    return (pred_bin & gm).sum() / (pred_bin | gm).sum()


def run_model(path, img):
    m = cpmodels.CellposeModel(gpu=True, pretrained_model=path)
    ca = 2 if (img.ndim == 3) else None
    return m.eval(img, channel_axis=ca, normalize=True)[0].astype(np.int32)


def fold_for_animal(manifest, animal):
    for f in manifest["folds"]:
        if animal in f["test_animals"]:
            return f["fold"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--models", required=True, help='comma list; "cpsam", a path, or "heldout:<patt{k}>"')
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    import json
    manifest = json.load(open(args.manifest))
    thr = args.iou

    segs = [p for d in args.data_dirs.split(",") for p in sorted(glob.glob(os.path.join(d.strip(), "*_seg.npy")))]
    segs = [p for p in segs if not AUG(clc_split.seg_stem(p))]
    print(f"images: {len(segs)}  IoU thr: {thr}\n", flush=True)

    # --- per image: load, split TP/FN, run cpsam, derive FP set ---
    data = []
    for sp in segs:
        img, masks, ism = load(sp)
        animal = clc_split.parse_animal(clc_split.seg_stem(sp))
        labels = label_ids(masks)
        man = ism[labels - 1]
        tp_labels = labels[~man]; fn_labels = labels[man]
        cp = run_model("cpsam", img)
        # FP = cpsam preds unmatched to any final GT mask
        fp_masks = []
        for l in label_ids(cp):
            pm = cp == l
            if best_iou_to_gt(pm, masks) < thr:
                fp_masks.append(pm)
        data.append(dict(sp=sp, img=img, masks=masks, animal=animal,
                         tp=tp_labels, fn=fn_labels, cp=cp, fp=fp_masks))
        print(f"  {os.path.basename(sp)[:42]:42s} animal={animal:6s} TP={len(tp_labels):3d} FN={len(fn_labels):2d} FP(cpsam)={len(fp_masks):3d}", flush=True)

    model_specs = [m.strip() for m in args.models.split(",") if m.strip()]

    def metrics_for(get_pred):
        rec=[]; sup=[]; ret=[]
        for d in data:
            pred = get_pred(d)
            for l in d["fn"]:
                rec.append(best_iou_to_pred(d["masks"] == l, pred) >= thr)
            for l in d["tp"]:
                ret.append(best_iou_to_pred(d["masks"] == l, pred) >= thr)
            for fpm in d["fp"]:
                sup.append(best_iou_to_pred(fpm, pred) < thr)  # suppressed = not reproduced
        return (np.mean(rec) if rec else float("nan"),
                np.mean(sup) if sup else float("nan"),
                np.mean(ret) if ret else float("nan"),
                len(rec), len(sup), len(ret))

    print(f"\n{'model':<40} {'FN-recov':>9} {'FP-supp':>8} {'TP-ret':>7}")
    print("-" * 70)
    for spec in model_specs:
        if spec == "cpsam":
            getp = lambda d: d["cp"]
        elif spec.startswith("heldout:"):
            patt = spec.split("heldout:", 1)[1]
            cache = {}
            def getp(d, patt=patt, cache=cache):
                k = fold_for_animal(manifest, d["animal"])
                path = patt.format(k=k)
                if (d["sp"], path) not in cache:
                    cache[(d["sp"], path)] = run_model(path, d["img"])
                return cache[(d["sp"], path)]
        else:
            cache = {}
            def getp(d, path=spec, cache=cache):
                if d["sp"] not in cache:
                    cache[d["sp"]] = run_model(path, d["img"])
                return cache[d["sp"]]
        fr, fs, tr, nr, ns, nt = metrics_for(getp)
        name = spec if len(spec) < 40 else "..." + spec[-37:]
        print(f"{name:<40} {fr:>8.1%} {fs:>8.1%} {tr:>7.1%}   (n FN={nr} FP={ns} TP={nt})", flush=True)


if __name__ == "__main__":
    main()
