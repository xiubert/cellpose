"""
Leave-one-animal-out (LOAO) cross-validation orchestration for CLC segmentation.

Two subcommands; the training itself runs per-fold via run_clc_cv.slurm (a SLURM
array that calls trainer_slurm.py then eval_seg.py for each fold).

  materialize  Build per-fold train/test dirs (symlinks) for a data recipe from
               the clc_split.py manifest. Each fold's test = the held-out
               animal's original images; train = the recipe's composition of the
               remaining animals (+ optional base dataset like Cunningham).

  aggregate    Pool the per-fold eval CSVs into one honest estimate: per-IMAGE AP
               pooled over all folds (NOT equal fold-averaging — 4 LOAO folds
               hold a single image), plus per-animal and per-age (adult/neonate)
               breakdowns, and an optional paired comparison vs a baseline CSV.

Recipes (train-set composition; test is always the held-out animal, originals):
  clc_all            all CLC train animals, both ages
  clc_adult          adult CLC train animals only; folds restricted to adult held-out
  clc_neonate        neonate CLC train animals only; folds restricted to neonate held-out
  cunningham_plus_clc  Cunningham train set + all CLC train animals (both ages)

Warm-start (init from label_xfer_aug_retest vs stock cpsam) is orthogonal to the
recipe — it's a training flag (CELLPOSE_PRETRAINED), set in run_clc_cv.slurm.

Usage:
  python clc_cv.py materialize --manifest clc_folds_loao.json --recipe clc_all \
      --out_root runs/clc_cv [--base_train DIR]   # base_train needed for cunningham_plus_clc
  python clc_cv.py aggregate --manifest clc_folds_loao.json --results_dir RUN_DIR \
      [--baseline_csv baseline.csv]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clc_split  # noqa: E402  (scan / seg_stem reuse)

_AGE_OF_RECIPE = {"clc_adult": "adult", "clc_neonate": "neonate"}


# ── materialize ───────────────────────────────────────────────────────────────

def _link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)


def _link_pair(item, dest_dir):
    """Symlink a sample's .tif + _seg.npy into dest_dir."""
    seg = item["path"]
    tif = os.path.join(os.path.dirname(seg), item["stem"] + ".tif")
    _link(seg, os.path.join(dest_dir, os.path.basename(seg)))
    if os.path.exists(tif):
        _link(tif, os.path.join(dest_dir, os.path.basename(tif)))


def materialize(args):
    with open(args.manifest) as f:
        manifest = json.load(f)
    items = clc_split.scan(manifest["root"], manifest["subdirs"])
    stem2item = {it["stem"]: it for it in items}

    recipe = args.recipe
    age_filter = _AGE_OF_RECIPE.get(recipe)          # None -> both ages
    use_base = recipe == "cunningham_plus_clc"
    if use_base and not args.base_train:
        raise SystemExit("recipe cunningham_plus_clc needs --base_train DIR")

    base_segs = []
    if use_base:
        base_segs = [p for p in sorted(glob.glob(os.path.join(args.base_train, "*_seg.npy")))
                     if not clc_split.is_aug(clc_split.seg_stem(p))]

    recipe_root = os.path.join(args.out_root, recipe)
    n_active = 0
    for fold in manifest["folds"]:
        # age-specific recipe: keep the fold but FILTER both test and train to the
        # recipe's age (so grouped mixed-age folds work, not just LOAO 1-animal
        # folds). Skip a fold only if it has no held-out animal of that age.
        test_items = [stem2item[s] for s in fold["test"]]
        if age_filter:
            test_items = [it for it in test_items if it["age"] == age_filter]
            if not test_items:
                continue
        n_active += 1
        fdir = os.path.join(recipe_root, f"fold_{fold['fold']}")
        train_dir = os.path.join(fdir, "train")
        test_dir = os.path.join(fdir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        train_items = [stem2item[s] for s in fold["train"]
                       if (age_filter is None or stem2item[s]["age"] == age_filter)]
        for it in train_items:
            _link_pair(it, train_dir)
        if use_base:
            for seg in base_segs:
                tif = os.path.join(os.path.dirname(seg), clc_split.seg_stem(seg) + ".tif")
                _link(seg, os.path.join(train_dir, os.path.basename(seg)))
                if os.path.exists(tif):
                    _link(tif, os.path.join(train_dir, os.path.basename(tif)))
        for it in test_items:
            _link_pair(it, test_dir)

        ntr = len(train_items) + len(base_segs)
        print(f"fold {fold['fold']}: test_animals={fold['test_animals']}  "
              f"train={ntr} (clc {len(train_items)}"
              + (f" + base {len(base_segs)}" if use_base else "") + f")  test={len(test_items)}")

    print(f"\nrecipe '{recipe}': {n_active} active folds -> {recipe_root}")
    if age_filter:
        print(f"  (age-restricted to held-out {age_filter} animals)")


# ── aggregate ────────────────────────────────────────────────────────────────

def _read_eval_csvs(results_dir):
    """Read all eval_*.csv under results_dir into per-image rows.

    Each row: dict(model, source, image, n_true, n_pred, AP@0.5, AP@0.75, AP@0.9).
    """
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(results_dir, "**", "eval_*.csv"), recursive=True)):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def _ap_cols(rows):
    return [c for c in rows[0] if c.startswith("AP@")] if rows else []


def _animal_age_maps(manifest):
    a2age = {a: d["age"] for a, d in manifest["animals"].items()}
    return a2age


def aggregate(args):
    with open(args.manifest) as f:
        manifest = json.load(f)
    a2age = _animal_age_maps(manifest)

    rows = _read_eval_csvs(args.results_dir)
    if not rows:
        raise SystemExit(f"no eval_*.csv under {args.results_dir}")
    ap_cols = _ap_cols(rows)

    # map image stem -> animal/age via clc_split parser
    def animal_of(stem):
        try:
            return clc_split.parse_animal(stem)
        except Exception:
            return "?"

    # Collapse per-fold model names (<recipe>_fold<k>) into ONE CV estimate:
    # each image was held out exactly once, so all fold rows pool into a single
    # cross-validated prediction set.
    def cv_model(m):
        return re.sub(r"_fold\d+$", "", m)

    for r in rows:
        r["model"] = cv_model(r["model"])
        r["animal"] = animal_of(r["image"])
        r["age"] = a2age.get(r["animal"], "?")
        for c in ap_cols:
            r[c] = float(r[c])

    models = sorted({r["model"] for r in rows})
    print(f"results_dir={args.results_dir}  {len(rows)} held-out predictions  "
          f"CV model(s): {models}\n")

    def pooled(rs):
        return {c: float(np.mean([r[c] for r in rs])) for c in ap_cols}

    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        print(f"=== {m}  (pooled over {len(mrows)} held-out images) ===")
        ov = pooled(mrows)
        print("  overall : " + "  ".join(f"{c}={ov[c]:.4f}" for c in ap_cols))
        for age in ("adult", "neonate"):
            ars = [r for r in mrows if r["age"] == age]
            if ars:
                av = pooled(ars)
                print(f"  {age:<8}: " + "  ".join(f"{c}={av[c]:.4f}" for c in ap_cols) + f"  (n={len(ars)})")
        print("  per-animal AP@0.5: " + ", ".join(
            f"{a}={np.mean([r[ap_cols[0]] for r in mrows if r['animal']==a]):.3f}"
            for a in sorted({r["animal"] for r in mrows})))
        print()

    # paired comparison vs baseline (per-image, same stems)
    if args.baseline_csv:
        with open(args.baseline_csv) as f:
            brows = list(csv.DictReader(f))
        base_models = sorted({r["model"] for r in brows})
        prim = ap_cols[0]
        for bm in base_models:
            b_by_img = {r["image"]: float(r[prim]) for r in brows if r["model"] == bm}
            for m in models:
                pairs = [(r[prim], b_by_img[r["image"]]) for r in rows
                         if r["model"] == m and r["image"] in b_by_img]
                if not pairs:
                    continue
                d = np.array([a - b for a, b in pairs])
                wins = int((d > 0).sum()); losses = int((d < 0).sum())
                print(f"paired {prim}: {m} vs {bm}  "
                      f"Δmean={d.mean():+.4f}  median={np.median(d):+.4f}  "
                      f"wins/losses/ties={wins}/{losses}/{len(d)-wins-losses}  (n={len(d)})")
                try:
                    from scipy.stats import wilcoxon
                    if np.any(d != 0):
                        stat, p = wilcoxon(d)
                        print(f"        Wilcoxon signed-rank p={p:.4g}")
                except Exception:
                    pass


def main():
    ap = argparse.ArgumentParser(description="LOAO CV orchestration for CLC segmentation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("materialize")
    m.add_argument("--manifest", required=True)
    m.add_argument("--recipe", required=True,
                   choices=["clc_all", "clc_adult", "clc_neonate", "cunningham_plus_clc"])
    m.add_argument("--out_root", default="runs/clc_cv")
    m.add_argument("--base_train", default=None, help="Cunningham train dir (for cunningham_plus_clc)")
    m.set_defaults(func=materialize)

    g = sub.add_parser("aggregate")
    g.add_argument("--manifest", required=True)
    g.add_argument("--results_dir", required=True)
    g.add_argument("--baseline_csv", default=None)
    g.set_defaults(func=aggregate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
