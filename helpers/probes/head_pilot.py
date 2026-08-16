"""
Feasibility pilot — is the hair-cell HEAD a visually separable sub-region of the
full-cell CLC mask? (Make-or-break for auto-converting full-cell GT -> head-only
without manual redraw; see notes/clc_seg_results.md §4½.)

No SAM here — this only answers "is there a distinct head to extract?" by:
  * detecting the labeled (MYO7A) channel,
  * sampling cells across the round(IHC)/oblong(OHC) spectrum,
  * rendering a montage: contrast-stretched cell crop + full-cell outline (red),
    + candidate head sub-regions from a within-mask Otsu split — darker mode
    (cyan) and brighter mode (yellow) — so we can see which matches the head,
  * printing per-cell separability metrics (within-mask intensity bimodality,
    Otsu contrast, dark/bright core area fraction & compactness).

Usage (in the cellpose container):
  python3 /helpers/head_pilot.py --seg <path_to_seg.npy> [--n 20] [--out montage.png]
"""

import argparse
import glob
import os

import numpy as np
import tifffile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label as cc_label


def seg_stem(p):
    b = os.path.basename(p)
    return b[:-len("_seg.npy")] if b.endswith("_seg.npy") else os.path.splitext(b)[0]


def load_img_masks(seg_path):
    d = np.load(seg_path, allow_pickle=True).item()
    masks = d["masks"]
    img = d.get("img")
    if img is None:
        tif = os.path.join(os.path.dirname(seg_path), seg_stem(seg_path) + ".tif")
        img = tifffile.imread(tif)
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 2, 3, 4) and img.shape[0] < img.shape[-1]:
        img = np.moveaxis(img, 0, -1)   # CHW -> HWC
    return img, masks


def labeled_channel(img, masks):
    """Channel whose in-mask vs out-of-mask mean contrast is largest = MYO7A."""
    if img.ndim == 2:
        return img.astype(np.float32)
    inside = masks > 0
    best, bestc = -np.inf, 0
    for c in range(img.shape[-1]):
        ch = img[..., c].astype(np.float32)
        contrast = ch[inside].mean() - ch[~inside].mean()
        if contrast > best:
            best, bestc = contrast, c
    return img[..., bestc].astype(np.float32)


def stretch(a, lo=1, hi=99):
    p1, p99 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / max(p99 - p1, 1e-6), 0, 1)


def bimodality_coeff(x):
    """Sarle's bimodality coefficient; >0.555 suggests bimodal/separable."""
    x = x.astype(np.float64)
    n = len(x)
    if n < 8 or x.std() == 0:
        return np.nan
    m = x.mean(); s = x.std()
    g = (((x - m) ** 3).mean()) / s**3          # skew
    k = (((x - m) ** 4).mean()) / s**4 - 3.0    # excess kurtosis
    return (g**2 + 1) / (k + 3.0 * (n - 1)**2 / ((n - 2) * (n - 3)))


def core_stats(crop_mask, vals, take_dark):
    """Otsu-split within mask; return (area_frac, compactness) of dark|bright core."""
    if vals.size < 8:
        return np.nan, np.nan
    try:
        thr = threshold_otsu(vals)
    except Exception:
        return np.nan, np.nan
    sel = np.zeros_like(crop_mask, dtype=bool)
    full = np.zeros_like(crop_mask, dtype=float)
    full[crop_mask] = vals
    sel[crop_mask] = (full[crop_mask] < thr) if take_dark else (full[crop_mask] >= thr)
    if sel.sum() == 0:
        return 0.0, np.nan
    lab = cc_label(sel)
    props = regionprops(lab)
    if not props:
        return 0.0, np.nan
    big = max(props, key=lambda p: p.area)
    area_frac = big.area / crop_mask.sum()
    # compactness: 4*pi*area / perimeter^2  (1=circle)
    per = big.perimeter or 1.0
    compact = min(1.0, 4 * np.pi * big.area / (per * per))
    return area_frac, compact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pad", type=int, default=8)
    args = ap.parse_args()

    img, masks = load_img_masks(args.seg)
    ch = labeled_channel(img, masks)
    print(f"image {img.shape}  masks {masks.shape}  cells {int(masks.max())}  MYO7A channel selected")

    props = regionprops(masks)
    props = [p for p in props if p.area >= 30]
    # stratified sample across eccentricity (round IHC -> oblong OHC)
    props.sort(key=lambda p: p.eccentricity)
    if len(props) > args.n:
        idx = np.linspace(0, len(props) - 1, args.n).round().astype(int)
        props = [props[i] for i in idx]

    chs = stretch(ch)
    ncol = 5
    nrow = int(np.ceil(len(props) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()

    print(f"\n{'cell':>5} {'area':>6} {'ecc':>5} {'bimod':>6} {'otsuC':>6} "
          f"{'darkAF':>6} {'darkCmp':>7} {'brtAF':>6} {'brtCmp':>7}")
    rows = []
    for ax, p in zip(axes, props):
        minr, minc, maxr, maxc = p.bbox
        minr = max(0, minr - args.pad); minc = max(0, minc - args.pad)
        maxr = min(masks.shape[0], maxr + args.pad); maxc = min(masks.shape[1], maxc + args.pad)
        crop = chs[minr:maxr, minc:maxc]
        cm = masks[minr:maxr, minc:maxc] == p.label
        vals = ch[minr:maxr, minc:maxc][cm]
        # normalize within-cell intensities for bimodality
        vn = (vals - vals.min()) / max(float(np.ptp(vals)), 1e-6)
        bc = bimodality_coeff(vn)
        try:
            otsu_c = (vals[vals >= threshold_otsu(vals)].mean() - vals[vals < threshold_otsu(vals)].mean()) / max(vals.std(), 1e-6)
        except Exception:
            otsu_c = np.nan
        dAF, dCmp = core_stats(cm, vals, take_dark=True)
        bAF, bCmp = core_stats(cm, vals, take_dark=False)
        rows.append((p.label, p.area, p.eccentricity, bc, otsu_c, dAF, dCmp, bAF, bCmp))
        print(f"{p.label:>5} {int(p.area):>6} {p.eccentricity:>5.2f} {bc:>6.2f} "
              f"{otsu_c:>6.2f} {dAF:>6.2f} {dCmp:>7.2f} {bAF:>6.2f} {bCmp:>7.2f}")

        ax.imshow(crop, cmap="magma")
        # full-cell outline (red)
        ax.contour(cm, levels=[0.5], colors="red", linewidths=1.2)
        # candidate head cores: dark (cyan), bright (yellow)
        full = np.zeros_like(cm, dtype=float); full[cm] = vals
        try:
            thr = threshold_otsu(vals)
            dark = cm & (full < thr); bright = cm & (full >= thr)
            for sel in (dark,):
                lab = cc_label(sel); pr = regionprops(lab)
                if pr:
                    big = (lab == max(pr, key=lambda q: q.area).label)
                    ax.contour(big, levels=[0.5], colors="cyan", linewidths=1.0)
            for sel in (bright,):
                lab = cc_label(sel); pr = regionprops(lab)
                if pr:
                    big = (lab == max(pr, key=lambda q: q.area).label)
                    ax.contour(big, levels=[0.5], colors="yellow", linewidths=1.0)
        except Exception:
            pass
        ax.set_title(f"#{p.label} ecc{p.eccentricity:.2f}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(props):]:
        ax.axis("off")

    fig.suptitle("CLC full-cell mask (red) vs intensity sub-regions: dark core (cyan) / bright core (yellow)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or (os.path.splitext(args.seg)[0] + "_head_pilot.png")
    fig.savefig(out, dpi=110)
    print(f"\nwrote montage: {out}")

    arr = np.array([r[3:] for r in rows], dtype=float)
    med = np.nanmedian(arr, axis=0)
    print(f"\nMEDIANS  bimod={med[0]:.2f} (>0.55 ~ separable)  otsuC={med[1]:.2f}  "
          f"darkAF={med[2]:.2f} darkCmp={med[3]:.2f}  brtAF={med[4]:.2f} brtCmp={med[5]:.2f}")


if __name__ == "__main__":
    main()
