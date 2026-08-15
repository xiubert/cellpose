"""Is fixed 0-255 scaling safe on the images that already work?
The deployed model was TRAINED under percentile normalisation, so this is a
covariate shift for the 92 normal images. A gain on 6 that costs the 92 is a
bad trade. Paired, deterministic, per-image."""
import glob, os
import numpy as np, tifffile
from cellpose import models
from cellpose.metrics import average_precision

MODEL = "/data/cellpose_cc/.cellpose/models/clc_all_full_deploy"
AUG = ("_SV_", "_rot90", "_rot180", "_rot270", "_fliph", "_flipv")
items = []
for d in ["/data/cellpose_cc/adult", "/data/cellpose_cc/neonate",
          "/data/cellpose_cc/adult(2)", "/data/cellpose_cc/neonate(2)"]:
    for s in sorted(glob.glob(os.path.join(d, "*_seg.npy"))):
        st = os.path.basename(s)[:-len("_seg.npy")]
        if any(a in st for a in AUG) or "20x" in st:
            continue
        t = os.path.join(d, st + ".tif")
        if os.path.exists(t):
            items.append((st, t, s))
print(f"{len(items)} images", flush=True)

model = models.CellposeModel(gpu=True, pretrained_model=MODEL)
rows = []
for st, t, s in items:
    gt = np.load(s, allow_pickle=True).item()["masks"].astype(np.int32)
    im = tifffile.imread(t)
    ca = 2 if im.ndim == 3 else None
    a = model.eval(im, channel_axis=ca, normalize=True)[0].astype(np.int32)
    x = im[..., 0] if im.ndim == 3 else im
    x = np.repeat(np.clip(x.astype(np.float32) / 255.0, 0, 1)[..., None], 3, axis=2)
    b = model.eval(x, channel_axis=2, normalize=False)[0].astype(np.int32)
    ap_a = float(average_precision(gt, a, threshold=[0.5])[0][0])
    ap_b = float(average_precision(gt, b, threshold=[0.5])[0][0])
    rows.append((int(gt.max()), ap_a, ap_b, st))
    print(f"  {int(gt.max()):5d} {ap_a:.3f} -> {ap_b:.3f}  {st[:40]}", flush=True)

import numpy as np
r = np.array([(x[0], x[1], x[2]) for x in rows])
dense = r[r[:, 0] >= 80]; sparse = r[r[:, 0] < 80]
print("\n================ SUMMARY ================")
for lab, sel in (("ALL", r), ("dense >=80 cells", dense), ("sparse <80", sparse)):
    if len(sel):
        d = sel[:, 2] - sel[:, 1]
        print(f"{lab:<18} n={len(sel):3d}  default={sel[:,1].mean():.4f}  "
              f"fixed={sel[:,2].mean():.4f}  Δ={d.mean():+.4f}  "
              f"W/T/L={int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())}")
try:
    from scipy.stats import wilcoxon
    d = r[:, 2] - r[:, 1]
    if np.any(d != 0):
        print(f"Wilcoxon (all): p={wilcoxon(r[:,2], r[:,1])[1]:.4g}")
except Exception:
    pass
