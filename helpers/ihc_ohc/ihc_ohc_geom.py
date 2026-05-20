"""
Phase 1 (geometric) — build a per-cell *geometric* feature table from
Cellpose _seg.npy files, the deterministic complement to the MYO7A CNN.

This mirrors ihc_ohc_crops.py: the *same* function
(`geom_features_for_seg`) builds the training table here and extracts
features at inference time (see ihc_ohc_geom_clf.py:predict_seg_geom).
Unlike the crop pipeline, features are inherently *whole-image* — the
cochlear-axis spline and the neighbour graph need every centroid — so the
unit of work is one seg file, not one isolated crop.

Why geometry classifies IHC vs OHC
----------------------------------
IHC form a single innermost row; OHC a ~3-wide band offset toward the
strial side. So the signed perpendicular distance of a cell to a smooth
curve fitted through *all* hair-cell centroids is the dominant signal, and
the IHC/OHC split is recoverable with no labels at all (the ~1:3 IHC:OHC
count prior + the single-file-vs-band geometry orient it — see
ihc_ohc_geom_clf.py `rule`). No training step, fully deterministic.

Feature groups (all per cell, names returned alongside the matrix)
-----------------------------------------------------------------
  shape    skimage.measure.regionprops on the instance label map —
           area, perimeter, eccentricity, solidity, extent, axis lengths,
           axis ratio, equivalent diameter. Unitless ones used raw; sizes
           normalised by the per-image median cell diameter (D) so the
           table is invariant to magnification.
  axis     cochlear centerline by *regression* — across-coordinate as a
           low-degree polynomial of the along-coordinate in PCA space (an
           interpolating spline zigzags across the 4-row band and kills the
           signal) → signed across-band offset w−g(t) (the key feature),
           |offset|, normalised along-position, tangent angle,
           cell-orientation-vs-axis, local curvature.
  nbr      cKDTree on centroids → mean/min kNN distance, count within 2D,
           neighbour-offset anisotropy (IHC row ≈ collinear → ~1; OHC band
           → lower), mean signed-perp of the k neighbours.
  edge     distance to the convex hull of all centroids (a no-annotation
           proxy for the tissue edge; secondary — kept but flagged).

Each cell also gets a `flag` (and a bitmask of reasons) marking
"this region looks weird" — high spline residual, sparse neighbourhood,
edge extrapolation, or too few cells in the image — for human review.

Labels come from the same source as the CNN: label_xfer.py's `class_map`
in the seg dict, else the original VOC XML (resolve_class_map, reused from
ihc_ohc_crops). Augmented D4 copies are skipped by default and grouped by
source image either way, so a cochlea never straddles the train/val split.

Dependencies (beyond the cellpose env): **scikit-image** (regionprops) and
scipy (already present). No torch — the geometric stack stays light.

Usage (inside the cellpose container; /data = /media/DATA/Chris/cellpose2D)
--------------------------------------------------------------------------
  python /helpers/ihc_ohc_geom.py \
      --data_dir /data/to_zip/hcat-data/Confocal/Cunningham/traintest/train \
      --out /helpers/geom_train.npz --preview /helpers/geom_train_preview.png
  python /helpers/ihc_ohc_geom.py \
      --data_dir /data/to_zip/hcat-data/Confocal/Cunningham/traintest/test \
      --out /helpers/geom_test.npz
"""

import argparse
import os

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from skimage.measure import regionprops_table

# Reuse the crop pipeline's seg-walking / label-resolution / grouping so the
# geometric and CNN tables are built over the *identical* set of cells and
# the *identical* leak-free image grouping.
from ihc_ohc_crops import (
    CLASS_NAMES, CLASS_TO_IDX, iter_seg_files, resolve_class_map, seg_stem,
)

# Bitmask reasons for the per-cell "looks weird" flag (≥1 → human review).
FLAG_SPLINE_RESID = 1   # cell sits far off the fitted cochlear axis
FLAG_SPARSE = 2         # too few neighbours (damaged / edge of tissue)
FLAG_EXTRAP = 4         # beyond a spline endpoint (extrapolated position)
FLAG_FEW_CELLS = 8      # whole image has too few cells to fit the axis

# Order matters: this is the column order of the returned feature matrix and
# of every saved table. Keep it stable so saved models stay loadable.
FEATURE_NAMES = [
    # shape (size features are /D; the rest are already unitless)
    "area_norm", "perim_norm", "eccentricity", "solidity", "extent",
    "major_norm", "minor_norm", "axis_ratio", "eqdiam_norm",
    # cochlear-axis
    "perp_signed", "perp_abs", "axis_pos", "tangent_angle",
    "orient_rel_axis", "curvature",
    # neighbourhood
    "knn_mean", "knn_min", "n_within_2d", "nbr_anisotropy", "nbr_perp_mean",
    # tissue-edge proxy (secondary)
    "hull_edge_dist",
]
N_FEATURES = len(FEATURE_NAMES)


# ── shape descriptors ───────────────────────────────────────────────────────────

def _shape_props(masks):
    """regionprops on the instance label map → dict keyed by mask id.

    One deterministic skimage call, no parameters. Returns, per label:
    centroid (row, col), and the raw shape descriptors. Sizes are
    normalised by the per-image median equivalent diameter by the caller.
    """
    t = regionprops_table(
        masks,
        properties=(
            "label", "area", "perimeter", "centroid",
            "eccentricity", "solidity", "extent",
            "major_axis_length", "minor_axis_length",
            "orientation", "equivalent_diameter_area",
        ),
    )
    props = {}
    for i, lab in enumerate(t["label"]):
        props[int(lab)] = dict(
            area=float(t["area"][i]),
            perimeter=float(t["perimeter"][i]),
            cy=float(t["centroid-0"][i]),      # row
            cx=float(t["centroid-1"][i]),      # col
            eccentricity=float(t["eccentricity"][i]),
            solidity=float(t["solidity"][i]),
            extent=float(t["extent"][i]),
            major=float(t["major_axis_length"][i]),
            minor=float(t["minor_axis_length"][i]),
            orientation=float(t["orientation"][i]),
            eqdiam=float(t["equivalent_diameter_area"][i]),
        )
    return props


# ── cochlear-axis spline ────────────────────────────────────────────────────────

def _fit_centerline(xs, ys):
    """Cochlear centerline as a regression, *not* an interpolating spline.

    The cochlea is a long thin organ: PC1 of the centroids runs along its
    length, PC2 across the IHC-row + 3-OHC-row band. Threading a spline
    through PC1-ordered points makes it zigzag across that ~4-wide band and
    sit ≈0 px from every cell, destroying the very signal we need. Instead
    fit the across-coordinate as a smooth low-degree polynomial of the
    along-coordinate, g(t): that follows the cochlea's gentle curvature but
    cannot bend to the row banding, so the residual w−g(t) *is* the
    IHC-vs-OHC offset (empirically separable at ~0.998 per image).

    Returns a dict: PCA `mean`/`pc1`/`pc2`, per-point `t` (along) and
    `w` (across), poly `coef`, `deg`, `rms` (centerline fit residual, for
    the spiral/fold flag), `ok`, and a `dense` xy polyline for QC.
    """
    pts = np.column_stack([xs, ys]).astype(float)
    n = len(pts)
    mean = pts.mean(0)
    cen = pts - mean
    if n < 4 or np.allclose(cen, 0):
        # degenerate: identity axis; caller flags FLAG_FEW_CELLS.
        e = np.array([1.0, 0.0])
        return dict(mean=mean, pc1=e, pc2=np.array([0.0, 1.0]),
                    t=cen[:, 0], w=cen[:, 1], coef=np.array([0.0]),
                    deg=0, rms=0.0, ok=False,
                    dense=np.repeat(mean[None], 2, 0))
    _, _, vt = np.linalg.svd(cen, full_matrices=False)
    pc1, pc2 = vt[0], vt[1]
    t = cen @ pc1                                     # along the cochlea
    w = cen @ pc2                                     # across the band
    # Degree grows with cell count: enough to track real cochlear curvature,
    # too low to chase the IHC/OHC banding. Capped at cubic.
    deg = 3 if n >= 12 else (2 if n >= 8 else (1 if n >= 5 else 0))
    coef = np.polyfit(t, w, deg)
    rms = float(np.sqrt(np.mean((w - np.polyval(coef, t)) ** 2)))
    ts = np.linspace(t.min(), t.max(), 256)
    dense = mean + np.outer(ts, pc1) + np.outer(np.polyval(coef, ts), pc2)
    return dict(mean=mean, pc1=pc1, pc2=pc2, t=t, w=w, coef=coef,
                deg=deg, rms=rms, ok=True, dense=dense)


def _axis_features(cl, orients, D):
    """Per-cell features relative to the fitted cochlear centerline.

    `perp_signed` = w − g(t), the across-band offset (PCA is orthonormal,
    so this is the perpendicular distance for a gently curved centerline) —
    the dominant IHC/OHC discriminator. Plus normalised along-position,
    local tangent angle, cell-orientation-vs-axis, local curvature, an
    outlier residual for the weird-cell flag, and an end-extrapolation
    mask. Distances are in pixels; the caller normalises by D.
    """
    t, w, coef = cl["t"], cl["w"], cl["coef"]
    g = np.polyval(coef, t)
    perp_signed = w - g
    perp_abs = np.abs(perp_signed)

    span = float(t.max() - t.min()) or 1.0
    axis_pos = (t - t.min()) / span                   # 0..1 along the cochlea

    # centerline tangent in image coords: d/dt[mean + t·pc1 + g(t)·pc2].
    gp = np.polyval(np.polyder(coef), t) if cl["deg"] >= 1 else np.zeros_like(t)
    gpp = np.polyval(np.polyder(coef, 2), t) if cl["deg"] >= 2 else np.zeros_like(t)
    tang = cl["pc1"][None, :] + gp[:, None] * cl["pc2"][None, :]
    tn = np.linalg.norm(tang, axis=1) + 1e-9
    tangent_angle = np.arctan2(tang[:, 1], tang[:, 0])
    # curvature of the parametric (t, g(t)) centerline: |x'×x''|/|x'|³.
    acc = gpp[:, None] * cl["pc2"][None, :]
    curv = np.abs(tang[:, 0] * acc[:, 1] - tang[:, 1] * acc[:, 0]) / tn ** 3

    rel = np.abs(orients - tangent_angle)
    orient_rel = np.minimum(rel % np.pi, np.pi - (rel % np.pi))

    # weird *cell*: across-offset far beyond the normal IHC+OHC envelope
    # (robust per-image scale; normal IHC is *expected* off-centre so this
    # keys on extreme outliers, e.g. a stray mask, not the IHC row itself).
    mad = np.median(np.abs(perp_abs - np.median(perp_abs))) + 1e-9
    resid = np.where(perp_abs > np.median(perp_abs) + 6.0 * mad,
                     perp_abs, 0.0)
    extrap = (axis_pos < 0.03) | (axis_pos > 0.97)    # cochlear-end risk
    return dict(perp_signed=perp_signed, perp_abs=perp_abs,
                axis_pos=axis_pos, tangent_angle=tangent_angle,
                orient_rel=orient_rel, curvature=curv,
                resid=resid, extrap=extrap)


# ── neighbourhood ───────────────────────────────────────────────────────────────

def _neighbour_features(xs, ys, perp_signed, D, k):
    """kNN-graph descriptors that separate a 1-D row from a 2-D band.

    `nbr_anisotropy` is the discriminator: the covariance of a cell's
    offset vectors to its k nearest neighbours is highly anisotropic for
    the single-file IHC row (≈1) and rounder for the OHC band (lower).
    """
    pts = np.column_stack([xs, ys]).astype(float)
    n = len(pts)
    tree = cKDTree(pts)
    kk = min(k, n - 1) if n > 1 else 0
    knn_mean = np.zeros(n)
    knn_min = np.zeros(n)
    aniso = np.zeros(n)
    nbr_perp = np.zeros(n)
    n_within = np.zeros(n)
    r = 2.0 * D
    for i in range(n):
        if kk == 0:
            continue
        d, j = tree.query(pts[i], k=kk + 1)        # includes self at d=0
        d, j = d[1:], j[1:]
        knn_mean[i] = d.mean()
        knn_min[i] = d.min()
        nbr_perp[i] = perp_signed[j].mean()
        off = pts[j] - pts[i]
        if len(off) >= 2:
            cov = np.cov(off.T)
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            aniso[i] = (ev[0] - ev[1]) / (ev[0] + ev[1] + 1e-9)
        n_within[i] = tree.query_ball_point(pts[i], r, return_length=True) - 1
    return dict(knn_mean=knn_mean, knn_min=knn_min, n_within=n_within,
                aniso=aniso, nbr_perp=nbr_perp)


def _hull_edge_dist(xs, ys):
    """Min distance from each centroid to the convex hull of all centroids.

    A no-annotation proxy for tissue-edge proximity (the design notes'
    fragile step): cells near the cochlear ends/edges sit close to the
    hull boundary. Secondary signal — kept but not relied upon.
    """
    pts = np.column_stack([xs, ys]).astype(float)
    n = len(pts)
    if n < 3:
        return np.zeros(n)
    try:
        hull = ConvexHull(pts)
    except Exception:  # noqa: BLE001 — collinear cells → no hull
        return np.zeros(n)
    verts = pts[hull.vertices]
    out = np.empty(n)
    for i, p in enumerate(pts):
        best = np.inf
        for a, b in zip(verts, np.roll(verts, -1, axis=0)):
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / (ab @ ab + 1e-9), 0, 1)
            best = min(best, np.linalg.norm(p - (a + t * ab)))
        out[i] = best
    return out


# ── single source of truth: features for one seg ────────────────────────────────

def geom_features_for_seg(seg, *, k_neighbors=6, min_cells=6):
    """Per-cell geometric features for one Cellpose seg dict.

    Returns (cell_ids, feats (N, N_FEATURES) float32, flags (N,) int,
    centroids (N, 2) xy) for *every* instance in `masks` (inference uses
    all; the builder keeps only labelled ones). Never raises on odd input
    — degenerate images fall back to a PC1 line and are flagged.
    """
    masks = seg["masks"]
    props = _shape_props(masks)
    cell_ids = sorted(props)
    n = len(cell_ids)
    if n == 0:
        return [], np.zeros((0, N_FEATURES), np.float32), \
            np.zeros(0, int), np.zeros((0, 2), np.float32)

    cx = np.array([props[c]["cx"] for c in cell_ids])
    cy = np.array([props[c]["cy"] for c in cell_ids])
    orient = np.array([props[c]["orientation"] for c in cell_ids])
    eqd = np.array([props[c]["eqdiam"] for c in cell_ids])
    D = float(np.median(eqd)) or 1.0                # per-image scale

    cl = _fit_centerline(cx, cy)
    ax = _axis_features(cl, orient, D)
    nb = _neighbour_features(cx, cy, ax["perp_signed"], D, k_neighbors)
    hull_d = _hull_edge_dist(cx, cy)

    feats = np.zeros((n, N_FEATURES), np.float32)
    for i, c in enumerate(cell_ids):
        p = props[c]
        feats[i] = [
            p["area"] / (D * D), p["perimeter"] / D, p["eccentricity"],
            p["solidity"], p["extent"], p["major"] / D, p["minor"] / D,
            p["minor"] / (p["major"] + 1e-9), p["eqdiam"] / D,
            ax["perp_signed"][i] / D, ax["perp_abs"][i] / D,
            ax["axis_pos"][i], ax["tangent_angle"][i],
            ax["orient_rel"][i], ax["curvature"][i] * D,
            nb["knn_mean"][i] / D, nb["knn_min"][i] / D, nb["n_within"][i],
            nb["aniso"][i], nb["nbr_perp"][i] / D, hull_d[i] / D,
        ]

    # weird-region flags (bitmask).
    flags = np.zeros(n, int)
    # per-cell across-band outlier (stray mask far off the organ of Corti)
    flags |= np.where(ax["resid"] > 0, FLAG_SPLINE_RESID, 0)
    # whole-image: centerline RMS ≫ a normal IHC+OHC band (~≤2·D) ⇒ the
    # cubic can't describe the layout (fold / tight spiral) — axis unreliable.
    if cl["rms"] > 3.0 * D:
        flags |= FLAG_SPLINE_RESID
    flags |= np.where(nb["n_within"] < 2, FLAG_SPARSE, 0)
    flags |= np.where(ax["extrap"], FLAG_EXTRAP, 0)
    if n < min_cells or not cl["ok"]:
        flags |= FLAG_FEW_CELLS

    cents = np.column_stack([cx, cy]).astype(np.float32)
    return cell_ids, feats, flags.astype(int), cents


# ── dataset builder ─────────────────────────────────────────────────────────────

def build_geom_dataset(data_dir, *, k_neighbors=6, include_augmented=False,
                        xml_dir=None):
    """Walk data_dir → (feats, labels, groups, cell_ids, flags, names).

    Same seg-walking / label-resolution / image-grouping as the crop
    builder, so the geometric and CNN tables cover the identical cells with
    the identical leak-free split.
    """
    F, Y, G, CID, FL, names = [], [], [], [], [], []
    gkey_to_idx = {}
    n_files = n_cells = n_skipped = 0
    src_counter = {"seg": 0, "xml": 0, "none": 0}

    for seg_path, gkey in iter_seg_files(data_dir, include_augmented):
        try:
            seg = np.load(seg_path, allow_pickle=True).item()
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR loading {os.path.basename(seg_path)}: {e}")
            continue
        if seg.get("masks") is None:
            print(f"  skip {os.path.basename(seg_path)} (no masks)")
            continue
        class_map, src = resolve_class_map(seg, seg_path, data_dir, xml_dir)
        src_counter[src] = src_counter.get(src, 0) + 1
        if not class_map:
            print(f"  skip {os.path.basename(seg_path)} (no class labels)")
            continue

        cell_ids, feats, flags, _ = geom_features_for_seg(
            seg, k_neighbors=k_neighbors)
        if gkey not in gkey_to_idx:
            gkey_to_idx[gkey] = len(names)
            names.append(seg_stem(seg_path))
        gidx = gkey_to_idx[gkey]

        kept = 0
        for cid, f, fl in zip(cell_ids, feats, flags):
            cls = class_map.get(cid)
            if cls not in CLASS_TO_IDX:
                n_skipped += 1
                continue
            F.append(f)
            Y.append(CLASS_TO_IDX[cls])
            G.append(gidx)
            CID.append(int(cid))
            FL.append(int(fl))
            kept += 1
        n_files += 1
        n_cells += kept
        print(f"  {seg_stem(seg_path):52s} {kept:4d} cells  grp{gidx:>3}  [{src}]")

    if not F:
        raise RuntimeError(f"no geometric features produced from {data_dir}")
    feats = np.stack(F).astype(np.float32)
    labels = np.asarray(Y, np.int64)
    n_ihc, n_ohc = int((labels == 0).sum()), int((labels == 1).sum())
    n_flag = int((np.asarray(FL) > 0).sum())
    print(f"\n{n_files} files, {len(names)} source images, {n_cells} cells "
          f"(IHC {n_ihc} / OHC {n_ohc}); {n_skipped} skipped; {n_flag} flagged")
    print(f"label source: {src_counter}")
    return (feats, labels, np.asarray(G, np.int64), np.asarray(CID, np.int64),
            np.asarray(FL, np.int64), names)


def save_geom_dataset(path, feats, labels, groups, cell_ids, flags, names,
                       meta):
    """Compressed .npz the geom classifier loads directly (parallels
    crops_*.npz; adds cell_ids so fusion can align with the CNN's
    per-cell class_prob)."""
    np.savez_compressed(
        path, feats=feats, labels=labels, groups=groups, cell_ids=cell_ids,
        flags=flags, image_names=np.asarray(names, dtype=object),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
        meta=np.asarray([meta], dtype=object))
    print(f"saved → {path}  {feats.shape}  "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")


def load_geom_npz(path):
    d = np.load(path, allow_pickle=True)
    meta = d["meta"][0] if "meta" in d else {}
    return dict(feats=d["feats"].astype(np.float32),
                labels=d["labels"].astype(np.int64),
                groups=d["groups"].astype(np.int64),
                cell_ids=d["cell_ids"].astype(np.int64),
                flags=d["flags"].astype(np.int64),
                image_names=list(d["image_names"]),
                feature_names=list(d["feature_names"]), meta=meta)


# ── QC overlay ──────────────────────────────────────────────────────────────────

def save_preview(path, data_dir, *, k_neighbors=6, max_images=12,
                 include_augmented=False, xml_dir=None):
    """Per-image overlay: centroids coloured by signed perp distance, the
    fitted axis, flagged cells ringed. Eyeball before trusting the table —
    this is where a folded spiral or a bad split shows itself."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("preview skipped (matplotlib not installed)")
        return

    segs = list(iter_seg_files(data_dir, include_augmented))[:max_images]
    cols = 3
    rows = (len(segs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for n, (seg_path, _) in enumerate(segs):
        seg = np.load(seg_path, allow_pickle=True).item()
        if seg.get("masks") is None:
            continue
        gt, _ = resolve_class_map(seg, seg_path, data_dir, xml_dir)
        cids, feats, flags, cents = geom_features_for_seg(
            seg, k_neighbors=k_neighbors)
        if len(cids) == 0:
            continue
        cl = _fit_centerline(cents[:, 0], cents[:, 1])
        ax = axes[n // cols][n % cols]
        ax.axis("on")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        dense = cl["dense"]
        ax.plot(dense[:, 0], dense[:, 1], "-", lw=1, color="0.5")
        perp = feats[:, FEATURE_NAMES.index("perp_signed")]
        sc = ax.scatter(cents[:, 0], cents[:, 1], c=perp, cmap="coolwarm",
                        s=14, vmin=-np.abs(perp).max(),
                        vmax=np.abs(perp).max())
        if gt:  # ring ground-truth IHC so the perp split is verifiable
            ihc = np.array([gt.get(c) == "IHC" for c in cids])
            ax.scatter(cents[ihc, 0], cents[ihc, 1], s=60,
                       facecolors="none", edgecolors="k", lw=0.6)
        bad = flags > 0
        if bad.any():
            ax.scatter(cents[bad, 0], cents[bad, 1], marker="x",
                       s=40, color="lime", lw=1)
        ax.set_title(f"{seg_stem(seg_path)[:28]}\n"
                     f"{len(cids)} cells, {int(bad.sum())} flagged",
                     fontsize=7)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("geom QC — colour=signed perp dist, ring=GT IHC, x=flagged",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"preview → {path}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build IHC/OHC geometric feature table from _seg.npy files")
    p.add_argument("--data_dir", required=True,
                   help="Directory of <stem>_myo7a_seg.npy files")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--k_neighbors", type=int, default=6,
                   help="k for the neighbour-graph features (default 6)")
    p.add_argument("--include-augmented", action="store_true",
                   help="Also ingest augment.py's on-disk D4 copies")
    p.add_argument("--xml_dir", default=None,
                   help="VOC XML dir (fallback when class_map is absent)")
    p.add_argument("--preview", default=None, metavar="PNG",
                   help="Also write a per-image QC overlay to this PNG")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Building geometric features from {args.data_dir}")
    feats, labels, groups, cell_ids, flags, names = build_geom_dataset(
        args.data_dir, k_neighbors=args.k_neighbors,
        include_augmented=args.include_augmented, xml_dir=args.xml_dir)
    meta = dict(k_neighbors=args.k_neighbors,
                data_dir=os.path.abspath(args.data_dir),
                feature_names=FEATURE_NAMES)
    save_geom_dataset(args.out, feats, labels, groups, cell_ids, flags,
                      names, meta)
    if args.preview:
        save_preview(args.preview, args.data_dir,
                     k_neighbors=args.k_neighbors,
                     include_augmented=args.include_augmented,
                     xml_dir=args.xml_dir)


if __name__ == "__main__":
    main()
