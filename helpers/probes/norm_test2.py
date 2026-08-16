"""Fixed ABSOLUTE scaling: use the dense-image regime's scale for every image,
so normalisation stops depending on how many cells happen to be in frame.
(The earlier normalize=False arm passed raw 0-255 and simply broke — that was
'no scaling', not 'fixed scaling'.)"""
import glob, os
import numpy as np, tifffile
from cellpose import models
from cellpose.metrics import average_precision

MODEL = "/data/cellpose_cc/.cellpose/models/clc_all_full_deploy"
imgs = []
for s in sorted(glob.glob("/data/cellpose_cc/adult(2)/*_seg.npy")):
    st = os.path.basename(s)[:-len("_seg.npy")]
    if not ("8477" in st or "8483" in st):
        continue
    m = np.load(s, allow_pickle=True).item()["masks"]
    im = tifffile.imread("/data/cellpose_cc/adult(2)/" + st + ".tif")
    imgs.append((st, im, m, int(m.max())))
imgs.sort(key=lambda r: r[3])

model = models.CellposeModel(gpu=True, pretrained_model=MODEL)
# dense-image reference scale measured earlier: p1~0, p99~239
REFS = [("fixed 0-239 (dense ref)", 0.0, 239.0),
        ("fixed 0-255 (full range)", 0.0, 255.0),
        ("fixed 0-150", 0.0, 150.0)]
print(f"{'config':<26} {'meanAP':>7}   per-image AP(n_pred/GT)")
for name, lo, hi in REFS:
    aps, ps = [], []
    for st, im, gt, n in imgs:
        x = im[..., 0] if im.ndim == 3 else im
        x = np.clip((x.astype(np.float32) - lo) / (hi - lo), 0, 1)
        x = np.repeat(x[..., None], 3, axis=2)
        pred = model.eval(x, channel_axis=2, normalize=False)[0].astype(np.int32)
        aps.append(float(average_precision(gt.astype(np.int32), pred, threshold=[0.5])[0][0]))
        ps.append(int(pred.max()))
    print(f"{name:<26} {np.mean(aps):>7.3f}   " +
          "  ".join(f"{a:.2f}({p}/{i[3]})" for a, p, i in zip(aps, ps, imgs)))
