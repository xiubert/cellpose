"""Does percentile normalisation break on sparse images?

Cellpose normalises each image to [p1, p99]. If cells occupy far less than 1% of
pixels, p99 lands in BACKGROUND, so the scaling is set by noise and the cells sit
far outside the normalised range. Test: compare p99 against the intensity
distribution INSIDE the ground-truth masks.
"""
import glob, os
import numpy as np, tifffile

rows = []
for d in ["/data/cellpose_cc/adult(2)", "/data/cellpose_cc/neonate(2)", "/data/cellpose_cc/adult"]:
    for s in sorted(glob.glob(os.path.join(d, "*_seg.npy"))):
        st = os.path.basename(s)[:-len("_seg.npy")]
        if "_SV_" in st:
            continue
        t = os.path.join(d, st + ".tif")
        if not os.path.exists(t):
            continue
        m = np.load(s, allow_pickle=True).item()["masks"]
        n = int(m.max())
        img = tifffile.imread(t)
        img = img[..., 0] if img.ndim == 3 else img
        img = img.astype(np.float32)
        frac = float((m > 0).mean())
        p1, p99 = np.percentile(img, [1, 99])
        cell_px = img[m > 0]
        if cell_px.size == 0:
            continue
        cell_med = float(np.median(cell_px))
        cell_p10 = float(np.percentile(cell_px, 10))
        # where does p99 sit within the cell intensity distribution?
        pct_of_cells_below_p99 = float((cell_px < p99).mean())
        rows.append((n, frac * 100, p1, p99, cell_med, cell_p10,
                     pct_of_cells_below_p99 * 100, st))

rows.sort()
print(f"{'cells':>5} {'%px':>6} {'p1':>5} {'p99':>6} {'cellmed':>8} {'cellp10':>8} "
      f"{'%cellpx<p99':>12}   image")
for n, frac, p1, p99, cm, cp10, below, st in rows:
    flag = "  <<< p99 is BELOW the typical cell" if p99 < cm else ""
    print(f"{n:>5} {frac:>6.2f} {p1:>5.0f} {p99:>6.0f} {cm:>8.0f} {cp10:>8.0f} "
          f"{below:>11.1f}%   {st[:34]}{flag}")

import numpy as np
sparse = [r for r in rows if r[0] < 80]
dense = [r for r in rows if r[0] >= 80]
for lab, sel in (("sparse (<80 cells)", sparse), ("dense (>=80)", dense)):
    if sel:
        print(f"\n{lab}: n={len(sel)}  median %px={np.median([r[1] for r in sel]):.2f}  "
              f"median p99={np.median([r[3] for r in sel]):.0f}  "
              f"median cell-median={np.median([r[4] for r in sel]):.0f}  "
              f"median %cellpx<p99={np.median([r[6] for r in sel]):.1f}%")
