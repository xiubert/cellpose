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
  podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py <cmd> ...

Run-artifact layout
-------------------
The full `train` pipeline writes three sibling timestamped dirs under
data.out_dir (default /helpers/ihc_ohc/runs/), all sharing one stamp so a
pipeline invocation groups visually:

  runs/20260520-104530_train-cnn/   best.pt, history.json, config.yaml, …
  runs/20260520-104530_train-geom/  geom_best.pkl, test_report.json, …
  runs/20260520-104530_fuse/        fuse.pkl, config.yaml

The pipeline echoes those paths at the end so subsequent `predict` calls
can use them (via --cnn_ckpt / --geom_ckpt / --fuse_ckpt).
"""

import argparse
import datetime
import os
import subprocess
import sys

import numpy as np
import yaml

HOST = "/helpers/ihc_ohc"          # container path of helpers/ihc_ohc/
CNN_CFG = f"{HOST}/configs/cnn.yaml"
GEOM_CFG = f"{HOST}/configs/geom.yaml"


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
    cache_default = f"{HOST}/runs/cache"
    crops_tr = _get(cnn, "data", "train_npz", f"{cache_default}/crops_train.npz")
    crops_te = _get(cnn, "data", "test_npz", f"{cache_default}/crops_test.npz")
    geom_tr = _get(geom, "data", "geom_train_npz", f"{cache_default}/geom_train.npz")
    geom_te = _get(geom, "data", "geom_test_npz", f"{cache_default}/geom_test.npz")

    # One shared timestamp for every step → all run dirs sit next to each
    # other under runs/ so a pipeline invocation is visually grouped.
    stamp = args.stamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cnn_out = _get(cnn, "data", "out_dir", f"{HOST}/runs")
    geom_out = _get(geom, "data", "out_dir", f"{HOST}/runs")
    cnn_run = f"{cnn_out}/{stamp}_train-cnn"
    geom_run = f"{geom_out}/{stamp}_train-geom"
    fuse_run = f"{geom_out}/{stamp}_fuse"
    cnn_ckpt = f"{cnn_run}/best.pt"
    os.makedirs(os.path.dirname(crops_tr), exist_ok=True)
    print(f"\n=== pipeline timestamp: {stamp} ===")

    py = sys.executable
    # 1. CNN crops → train CNN
    _run([py, f"{HOST}/ihc_ohc_crops.py", "--data_dir", train_dir,
          "--out", crops_tr,
          "--preview", f"{cache_default}/crops_train_preview.png"])
    _run([py, f"{HOST}/ihc_ohc_crops.py", "--data_dir", test_dir,
          "--out", crops_te])
    _run([py, f"{HOST}/ihc_ohc_classifier.py", "train",
          "--config", args.cnn_config, "--run_dir", cnn_run])
    # 2. geom table → train geom
    _run([py, f"{HOST}/ihc_ohc_geom.py", "--data_dir", train_dir,
          "--out", geom_tr,
          "--preview", f"{cache_default}/geom_train_preview.png"])
    _run([py, f"{HOST}/ihc_ohc_geom.py", "--data_dir", test_dir,
          "--out", geom_te])
    _run([py, f"{HOST}/ihc_ohc_geom_clf.py", "train",
          "--config", args.geom_config, "--run_dir", geom_run])
    # 3. CNN probs into every seg (fusion prerequisite) → fuse
    print("\n=== writing CNN class_prob into all segs (fusion prereq) ===")
    cnn_predict_dir(cnn_ckpt, train_dir)
    cnn_predict_dir(cnn_ckpt, test_dir)
    _run([py, f"{HOST}/ihc_ohc_geom_clf.py", "fuse",
          "--config", args.geom_config, "--run_dir", fuse_run])
    print("\n✓ pipeline complete:")
    print(f"   CNN  ckpt → {cnn_ckpt}")
    print(f"   geom ckpt → {geom_run}/geom_best.pkl")
    print(f"   fuse ckpt → {fuse_run}/fuse.pkl")


# ── predict: score a seg / dir with the whole stack ─────────────────────────────

def cmd_predict(args):
    """Score a seg / dir with the full stack.

    With timestamped run dirs there is no canonical "latest" path, so the
    three checkpoint paths must be passed explicitly (CLI or
    data.cnn_ckpt / data.geom_ckpt / data.fuse_ckpt in the YAML configs).
    """
    cnn = _yaml(args.cnn_config)
    geom = _yaml(args.geom_config)
    cnn_ckpt = args.cnn_ckpt or _get(cnn, "data", "cnn_ckpt", None)
    geom_ckpt = args.geom_ckpt or _get(geom, "data", "geom_ckpt", None)
    fuse_ckpt = args.fuse_ckpt or _get(geom, "data", "fuse_ckpt", None)
    missing = [n for n, v in (("--cnn_ckpt", cnn_ckpt),
                              ("--geom_ckpt", geom_ckpt),
                              ("--fuse_ckpt", fuse_ckpt)) if not v]
    if missing:
        sys.exit(f"predict: missing checkpoint path(s) {missing}. Pass via "
                 "CLI flag or set data.cnn_ckpt / geom_ckpt / fuse_ckpt in "
                 "the YAML configs (each train/fuse run prints its run dir).")
    py = sys.executable

    if args.dir:
        from ihc_ohc_crops import iter_seg_files
        print(f"=== CNN write-back over {args.dir} ===")
        cnn_predict_dir(cnn_ckpt, args.dir)
        for sp, _ in iter_seg_files(args.dir):
            _run([py, f"{HOST}/ihc_ohc_geom_clf.py", "predict",
                  "--geom_ckpt", geom_ckpt, "--fuse_ckpt", fuse_ckpt,
                  "--fuse", "--seg", sp, "--write"])
    else:
        # single seg: the two per-seg CLIs as-is (single source of truth)
        _run([py, f"{HOST}/ihc_ohc_classifier.py", "predict",
              "--ckpt", cnn_ckpt, "--seg", args.seg, "--write"])
        _run([py, f"{HOST}/ihc_ohc_geom_clf.py", "predict",
              "--geom_ckpt", geom_ckpt, "--fuse_ckpt", fuse_ckpt,
              "--fuse", "--seg", args.seg, "--write"])


# ── migrate-preds: move legacy in-seg prediction keys to sidecars ───────────────

def cmd_migrate_preds(args):
    """One-time housekeeping for segs from before the sidecar convention.

    For each `_seg.npy` in `--dir`: any prediction keys still living in
    the seg dict are moved to `<stem>_pred.npy` (without clobbering values
    already in an existing sidecar — the sidecar wins on conflict) and
    then *removed* from the seg, which is re-saved clean. This is the
    only operation that writes to a dataset seg file, and it's opt-in.
    Use `--dry-run` to see what would change first.
    """
    from ihc_ohc_crops import (
        PRED_KEYS, iter_seg_files, pred_path, update_pred,
    )
    n_segs = n_keys = n_skip = n_clean = 0
    for sp, _ in iter_seg_files(args.dir, include_augmented=args.include_augmented):
        seg = np.load(sp, allow_pickle=True).item()
        in_seg = [k for k in PRED_KEYS if k in seg]
        if not in_seg:
            n_clean += 1
            continue
        # Read the actual sidecar file (NOT load_pred — that has a legacy
        # seg-fallback which would falsely report keys "already present"
        # and let us strip them from seg without persisting → data loss).
        pp = pred_path(sp)
        existing = {}
        if os.path.exists(pp):
            try:
                existing = np.load(pp, allow_pickle=True).item() or {}
            except Exception:  # noqa: BLE001
                existing = {}
        # Migrate every in-seg key not already in the real sidecar.
        to_move = {k: seg[k] for k in in_seg if k not in existing}
        already = [k for k in in_seg if k in existing]
        print(f"  {os.path.basename(sp)}: seg has {in_seg}; "
              f"migrating {list(to_move)}"
              + (f"; sidecar already has {already}" if already else ""))
        n_segs += 1
        n_keys += len(to_move)
        n_skip += len(already)
        if args.dry_run:
            continue
        # Only after the sidecar write *succeeds* may we pop from seg.
        if to_move:
            update_pred(sp, **to_move)
        for k in in_seg:
            seg.pop(k, None)
        np.save(sp, seg)
    print(f"\n{n_segs} segs migrated  |  {n_keys} keys moved  |  "
          f"{n_skip} keys skipped (sidecar wins)  |  "
          f"{n_clean} segs already clean"
          + ("  [DRY RUN — no files written]" if args.dry_run else ""))
    if not args.dry_run and n_segs:
        print(f"  → sidecars written; legacy in-seg keys removed; "
              f"check one with: python3 -c 'import numpy as np; "
              f"print(sorted(np.load(\"<seg>\",allow_pickle=True).item().keys()))'")


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
    t.add_argument("--stamp", default=None,
                   help="timestamp prefix shared by every run dir in this "
                   "pipeline invocation (default: %%Y%%m%%d-%%H%%M%%S now)")

    q = sub.add_parser("predict", help="score a seg / dir with the full stack")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--seg", help="single _seg.npy")
    g.add_argument("--dir", help="directory of _seg.npy (batched)")
    q.add_argument("--cnn_ckpt", default=None)
    q.add_argument("--geom_ckpt", default=None)
    q.add_argument("--fuse_ckpt", default=None)
    q.add_argument("--cnn_config", default=CNN_CFG)
    q.add_argument("--geom_config", default=GEOM_CFG)

    m = sub.add_parser(
        "migrate-preds",
        help="move legacy in-seg prediction keys to sidecars (one-time housekeeping)")
    m.add_argument("--dir", required=True, help="dir of _seg.npy to clean")
    m.add_argument("--dry-run", action="store_true",
                   help="report what would change; don't write anything")
    m.add_argument("--include-augmented", action="store_true",
                   help="also process augment.py's D4 copies")

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
    {"train": cmd_train, "predict": cmd_predict, "plot": cmd_plot,
     "migrate-preds": cmd_migrate_preds}[args.cmd](args)


if __name__ == "__main__":
    main()
