"""Does changing normalisation recover the degeneration images? Inference only."""
import glob, os
import numpy as np, tifffile
from cellpose import models
from cellpose.metrics import average_precision

MODEL = "/data/cellpose_cc/.cellpose/models/clc_all_full_deploy"
imgs = []
for d in ["/data/cellpose_cc/adult(2)"]:
    for s in sorted(glob.glob(os.path.join(d, "*_seg.npy"))):
        st = os.path.basename(s)[:-len("_seg.npy")]
        if not ("8477" in st or "8483" in st):
            continue
        m = np.load(s, allow_pickle=True).item()["masks"]
        im = tifffile.imread(os.path.join(d, st + ".tif"))
        imgs.append((st, im, m, int(m.max())))
imgs.sort(key=lambda r: r[3])
print(f"{len(imgs)} degeneration images\n")

CONFIGS = [
    ("default [1,99]",        dict(normalize=True)),
    ("percentile [1,99.99]",  dict(normalize={"percentile": [1.0, 99.99]})),
    ("percentile [0.1,99.9]", dict(normalize={"percentile": [0.1, 99.9]})),
    ("no normalisation",      dict(normalize=False)),
]
model = models.CellposeModel(gpu=True, pretrained_model=MODEL)
res = {}
for name, kw in CONFIGS:
    aps, npreds = [], []
    for st, im, gt, n in imgs:
        ca = 2 if im.ndim == 3 else None
        try:
            pred = model.eval(im, channel_axis=ca, **kw)[0].astype(np.int32)
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}: {e}"); aps = None; break
        ap = average_precision(gt.astype(np.int32), pred, threshold=[0.5])[0][0]
        aps.append(float(ap)); npreds.append(int(pred.max()))
    if aps is None:
        continue
    res[name] = (aps, npreds)
    print(f"{name:<24} mean AP@0.5={np.mean(aps):.3f}   " +
          "  ".join(f"{a:.2f}({p}/{i[3]})" for a, p, i in zip(aps, npreds, imgs)))
print("\nper-image detail (GT cells / AP by config):")
print(f"{'GT':>5} " + " ".join(f"{n[:16]:>17}" for n in res))
for j, (st, im, gt, n) in enumerate(imgs):
    print(f"{n:>5} " + " ".join(f"{res[k][0][j]:>9.3f}({res[k][1][j]:>3d})" for k in res))
