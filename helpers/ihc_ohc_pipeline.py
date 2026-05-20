"""
End-to-end IHC/OHC orchestrator — one entry point for the whole stack
(CNN + geometric + fusion) plus a class-coloured mask plotter.

This is a *thin* driver: every learning step shells out to the existing,
already-tested CLIs so each module stays the single source of truth. The
only logic added here is (a) the batched CNN write-back over a directory
(the one step that isn't a single CLI call — fusion needs the CNN's
class_prob in every seg) and (b) the plot.

Subcommands
-----------
  train    full pipeline: build CNN crops → train CNN → build geom table
           → train geom → write CNN probs into every seg → fuse. Produces
           best.pt, geom_best.pkl, fuse.pkl + the held-out test report.

  predict  score a single _seg.npy (or every seg in a dir) with the whole
           stack and write class_map_pred / class_prob (CNN),
           class_map_geom / class_prob_geom / geom_flag, and
           class_map_fused back in — non-destructively (masks untouched).

  plot     render one image's masks tinted by IHC/OHC, from any source
           (fused | geom | cnn | gt); flagged cells ringed. → PNG.

Run inside the cellpose container (paths are container paths):
  podman exec cellpose python3 /helpers/ihc_ohc_pipeline.py <cmd> ...
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import yaml

HELP = "/helpers"
CNN_CFG = f"{HELP}/ihc_ohc_config.yaml"
GEOM_CFG = f"{HELP}/ihc_ohc_geom_config.yaml"


# ── config helpers (read-only; defaults mirror each module's DEFAULT_CONFIG) ─────

def _yaml(path):
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def _get(cfg, section, key, default):
    return (cfg.get(section) or {}).get(key, default)


def _run(cmd):
    """Echo + run a CLI step; abort the pipeline on failure."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"step failed ({r.returncode}): {' '.join(cmd)}")


# ── the one non-CLI step: batched CNN write-back over a directory ────────────────

def cnn_predict_dir(ckpt, data_dir, *, include_augmented=False):
    """Write class_map_pred / class_prob into every seg in `data_dir`.

    Mirrors ihc_ohc_classifier.predict_seg but loads the model once and
    batches the directory (≈10× faster than looping the per-seg CLI) —
    fusion needs the CNN prob present in every seg first.
    """
    import torch  # lazy: keeps `plot` torch-free

    from ihc_ohc_classifier import CLASS_NAMES, load_model, predict_crops
    from ihc_ohc_crops import (
        extract_cell_crop, iter_seg_files, load_image_plane, tif_for_seg,
        update_pred,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(ckpt, dev)
    cm = ck.get("crop_meta", {}) or {}
    osz = int(cm.get("out_size", 64))
    pf = float(cm.get("pad_frac", 0.5))
    pp = cm.get("pad_px")
    sm = bool(cm.get("soft_mask", True))
    chn = int(cm.get("channel", 1))

    n = 0
    for sp, _ in iter_seg_files(data_dir, include_augmented):
        seg = np.load(sp, allow_pickle=True).item()
        masks = seg.get("masks")
        if masks is None:
            continue
        plane = load_image_plane(seg, tif_for_seg(sp), chn)
        fill = float(plane.mean())
        ids = [int(i) for i in np.unique(masks) if i != 0]
        crops, valid = [], []
        for cid in ids:
            cr = extract_cell_crop(plane, masks, cid, out_size=osz,
                                   pad_frac=pf, pad_px=pp, pad_value=fill,
                                   soft_mask=sm)
            if cr is not None:
                crops.append(cr)
                valid.append(cid)
        if not crops:
            continue
        pr, pb = predict_crops(model, np.stack(crops), ck["mean"],
                               ck["std"], dev)
        update_pred(
            sp,
            class_map_pred={c: CLASS_NAMES[p] for c, p in zip(valid, pr)},
            class_prob={c: float(pb_[p])
                        for c, p, pb_ in zip(valid, pr, pb)})
        n += 1
    print(f"  CNN write-back: {n} sidecars (seg files untouched) in {data_dir}")


# ── train: the full pipeline ────────────────────────────────────────────────────

def cmd_train(args):
    cnn = _yaml(args.cnn_config)
    geom = _yaml(args.geom_config)
    train_dir, test_dir = args.train_dir, args.test_dir
    crops_tr = _get(cnn, "data", "train_npz", f"{HELP}/crops_train.npz")
    crops_te = _get(cnn, "data", "test_npz", f"{HELP}/crops_test.npz")
    cnn_ckpt = os.path.join(
        _get(cnn, "data", "out_dir", f"{HELP}/ihc_ohc_run"), "best.pt")
    geom_tr = _get(geom, "data", "geom_train_npz", f"{HELP}/geom_train.npz")
    geom_te = _get(geom, "data", "geom_test_npz", f"{HELP}/geom_test.npz")

    py = sys.executable
    # 1. CNN crops → train CNN
    _run([py, f"{HELP}/ihc_ohc_crops.py", "--data_dir", train_dir,
          "--out", crops_tr, "--preview", f"{HELP}/crops_train_preview.png"])
    _run([py, f"{HELP}/ihc_ohc_crops.py", "--data_dir", test_dir,
          "--out", crops_te])
    _run([py, f"{HELP}/ihc_ohc_classifier.py", "train",
          "--config", args.cnn_config])
    # 2. geom table → train geom
    _run([py, f"{HELP}/ihc_ohc_geom.py", "--data_dir", train_dir,
          "--out", geom_tr, "--preview", f"{HELP}/geom_train_preview.png"])
    _run([py, f"{HELP}/ihc_ohc_geom.py", "--data_dir", test_dir,
          "--out", geom_te])
    _run([py, f"{HELP}/ihc_ohc_geom_clf.py", "train",
          "--config", args.geom_config])
    # 3. CNN probs into every seg (fusion prerequisite) → fuse
    print("\n=== writing CNN class_prob into all segs (fusion prereq) ===")
    cnn_predict_dir(cnn_ckpt, train_dir)
    cnn_predict_dir(cnn_ckpt, test_dir)
    _run([py, f"{HELP}/ihc_ohc_geom_clf.py", "fuse",
          "--config", args.geom_config])
    print("\n✓ pipeline complete — best.pt, geom_best.pkl, fuse.pkl ready")


# ── predict: score a seg / dir with the whole stack ─────────────────────────────

def cmd_predict(args):
    cnn = _yaml(args.cnn_config)
    geom = _yaml(args.geom_config)
    cnn_ckpt = args.cnn_ckpt or os.path.join(
        _get(cnn, "data", "out_dir", f"{HELP}/ihc_ohc_run"), "best.pt")
    out_dir = _get(geom, "data", "out_dir", f"{HELP}/ihc_ohc_geom_run")
    geom_ckpt = args.geom_ckpt or f"{out_dir}/geom_best.pkl"
    fuse_ckpt = args.fuse_ckpt or f"{out_dir}/fuse.pkl"
    py = sys.executable

    if args.dir:
        from ihc_ohc_crops import iter_seg_files
        print(f"=== CNN write-back over {args.dir} ===")
        cnn_predict_dir(cnn_ckpt, args.dir)
        for sp, _ in iter_seg_files(args.dir):
            _run([py, f"{HELP}/ihc_ohc_geom_clf.py", "predict",
                  "--geom_ckpt", geom_ckpt, "--fuse_ckpt", fuse_ckpt,
                  "--fuse", "--seg", sp, "--write"])
    else:
        # single seg: the two per-seg CLIs as-is (single source of truth)
        _run([py, f"{HELP}/ihc_ohc_classifier.py", "predict",
              "--ckpt", cnn_ckpt, "--seg", args.seg, "--write"])
        _run([py, f"{HELP}/ihc_ohc_geom_clf.py", "predict",
              "--geom_ckpt", geom_ckpt, "--fuse_ckpt", fuse_ckpt,
              "--fuse", "--seg", args.seg, "--write"])


# ── plot: masks tinted by IHC/OHC ───────────────────────────────────────────────

# IHC = warm, OHC = cool, unlabelled = grey. RGB, 0–1.
_CLASS_RGB = {"IHC": (0.95, 0.25, 0.20), "OHC": (0.15, 0.65, 0.95)}
_SOURCE_KEY = {"fused": "class_map_fused", "geom": "class_map_geom",
               "cnn": "class_map_pred", "gt": "class_map"}


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from ihc_ohc_crops import (
        load_image_plane, load_pred, resolve_class_map, tif_for_seg,
    )

    seg = np.load(args.seg, allow_pickle=True).item()
    masks = seg["masks"]
    plane = load_image_plane(seg, tif_for_seg(args.seg))
    pred = load_pred(args.seg, seg)

    key = _SOURCE_KEY[args.source]
    if args.source == "gt":
        cmap, _ = resolve_class_map(seg, args.seg, os.path.dirname(args.seg))
    else:
        cmap = {int(k): v for k, v in (pred.get(key) or {}).items()}
    if not cmap:
        sys.exit(f"no '{key}' for {os.path.basename(args.seg)} — run "
                 f"`ihc_ohc_pipeline.py predict` first (source={args.source})")
    flags = {int(k): int(v) for k, v in (pred.get("geom_flag") or {}).items()}

    # grayscale MYO7A background, robust percentile stretch.
    lo, hi = np.percentile(plane, (1, 99.5))
    bg = np.clip((plane - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = np.dstack([bg, bg, bg])

    counts = {"IHC": 0, "OHC": 0}
    overlay = np.zeros_like(rgb)
    alpha = np.zeros(masks.shape, np.float32)
    for cid, cls in cmap.items():
        if cls not in _CLASS_RGB:
            continue
        m = masks == cid
        overlay[m] = _CLASS_RGB[cls]
        alpha[m] = args.alpha
        counts[cls] += 1
    rgb = rgb * (1 - alpha[..., None]) + overlay * alpha[..., None]

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(rgb)
    ax.axis("off")

    # ring flagged cells (geometry uncertain → human review)
    n_flag = 0
    if args.show_flags and flags:
        for cid, fl in flags.items():
            if fl <= 0 or cid not in cmap:
                continue
            ys, xs = np.where(masks == cid)
            if ys.size:
                ax.add_patch(mpatches.Circle(
                    (xs.mean(), ys.mean()),
                    radius=0.6 * max(np.ptp(xs), np.ptp(ys), 6) + 4,
                    fill=False, edgecolor="yellow", lw=1.2))
                n_flag += 1

    acc = ""
    gt, src = resolve_class_map(seg, args.seg, os.path.dirname(args.seg))
    if gt and args.source != "gt":
        ok = sum(gt.get(c) == v for c, v in cmap.items() if c in gt)
        tot = sum(c in gt for c in cmap)
        if tot:
            acc = f"  |  vs GT [{src}] acc {ok/tot:.3f} ({ok}/{tot})"

    stem = os.path.basename(args.seg).replace("_seg.npy", "")
    ax.set_title(f"{stem}  |  source={args.source}  |  "
                 f"IHC {counts['IHC']} / OHC {counts['OHC']}"
                 + (f"  |  {n_flag} flagged" if n_flag else "") + acc,
                 fontsize=11)
    leg = [mpatches.Patch(color=_CLASS_RGB[c], label=c) for c in ("IHC", "OHC")]
    if n_flag:
        leg.append(mpatches.Patch(edgecolor="yellow", facecolor="none",
                                  label="flagged"))
    ax.legend(handles=leg, loc="upper right", framealpha=0.6)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.seg)),
        f"{stem}_{args.source}_ihc_ohc.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="IHC/OHC end-to-end pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="full pipeline → all checkpoints")
    t.add_argument("--train_dir", required=True, help="seg+tif dir (train)")
    t.add_argument("--test_dir", required=True, help="seg+tif dir (test)")
    t.add_argument("--cnn_config", default=CNN_CFG)
    t.add_argument("--geom_config", default=GEOM_CFG)

    q = sub.add_parser("predict", help="score a seg / dir with the full stack")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--seg", help="single _seg.npy")
    g.add_argument("--dir", help="directory of _seg.npy (batched)")
    q.add_argument("--cnn_ckpt", default=None)
    q.add_argument("--geom_ckpt", default=None)
    q.add_argument("--fuse_ckpt", default=None)
    q.add_argument("--cnn_config", default=CNN_CFG)
    q.add_argument("--geom_config", default=GEOM_CFG)

    v = sub.add_parser("plot", help="masks tinted by IHC/OHC → PNG")
    v.add_argument("--seg", required=True)
    v.add_argument("--source", default="fused",
                   choices=["fused", "geom", "cnn", "gt"],
                   help="which label to colour by (default fused)")
    v.add_argument("--out", default=None, help="PNG path (default next to seg)")
    v.add_argument("--alpha", type=float, default=0.45,
                   help="mask tint opacity (default 0.45)")
    v.add_argument("--dpi", type=int, default=150)
    v.add_argument("--no-flags", dest="show_flags", action="store_false",
                   help="don't ring geometry-flagged cells")
    return p.parse_args()


def main():
    args = parse_args()
    {"train": cmd_train, "predict": cmd_predict,
     "plot": cmd_plot}[args.cmd](args)


if __name__ == "__main__":
    main()
