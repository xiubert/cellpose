"""Fixed-scale arm vs the clc_all_full3 control, paired on identical folds.

Both arms: same 98 images, same v4 folds, same recipe. The ONLY difference is
the input representation — fixed 0-255 scaling with cellpose's per-image
[p1,p99] normalisation disabled, vs stock percentile normalisation.
"""
import csv, glob, os
import numpy as np
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def pool(d):
    ap = {}
    for f in sorted(glob.glob(os.path.join(d, "eval_fold*.csv"))):
        for r in csv.DictReader(open(f)):
            ap[r["image"]] = (float(r["AP@0.5"]), float(r["n_true"]), float(r["n_pred"]))
    return ap


F = pool("run_logs/clc_all_fixedscale_20260815")   # fixed scale
D = pool("run_logs/clc_all_full3_20260814")        # control, same folds/data
keys = sorted(set(F) & set(D))
print(f"paired images: {len(keys)}  (fixed {len(F)}, control {len(D)})\n")


def blk(label, ks):
    if not ks:
        return
    f = np.array([F[k][0] for k in ks])
    d = np.array([D[k][0] for k in ks])
    diff = f - d
    line = (f"  {label:<26s} n={len(ks):3d} fixed={f.mean():.4f} ctrl={d.mean():.4f} "
            f"Δ={diff.mean():+.4f} "
            f"W/T/L={int((diff > 0).sum())}/{int((diff == 0).sum())}/{int((diff < 0).sum())}")
    if wilcoxon is not None and np.any(diff != 0):
        try:
            line += f" p={wilcoxon(f, d)[1]:.4g}"
        except Exception:
            pass
    print(line)


def sparse(k):
    return D[k][1] < 80


blk("ALL", keys)
blk("dense (>=80 GT cells)", [k for k in keys if not sparse(k)])
blk("sparse (<80)", [k for k in keys if sparse(k)])

print("\n  per-image, sparse subset (GT / ctrl / fixed):")
for k in sorted([k for k in keys if sparse(k)], key=lambda k: D[k][1]):
    print(f"    GT={int(D[k][1]):3d}  ctrl={D[k][0]:.3f}({int(D[k][2]):3d})  "
          f"fixed={F[k][0]:.3f}({int(F[k][2]):3d})   {k[42:76]}")

sw = sorted(keys, key=lambda k: F[k][0] - D[k][0])
print("\n  biggest swings (fixed - ctrl):")
for k in sw[:4] + sw[-4:]:
    print(f"    {F[k][0] - D[k][0]:+.3f}  (GT={int(D[k][1]):4d})  {k[:54]}")
