"""
Phase 2 (geometric) — classify IHC vs OHC from the geometric feature
table, and *fuse* with the MYO7A CNN (Path A: late / ensemble fusion).

Why an ensemble. The CNN (ihc_ohc_classifier.py) reads MYO7A appearance;
this reads pure geometry. Their errors are largely independent — the CNN
struggles on faint/odd-staining cells the geometry places unambiguously,
the geometry struggles in damaged regions the CNN still reads from
texture. Averaging two decorrelated 0.9-ish classifiers is the cheapest
real accuracy gain available, and it needs no change to the tuned CNN: it
already writes `class_prob` into the seg dict expressly for this.

Subcommands (each takes --config <yaml>, except `predict`)
----------------------------------------------------------
  rule     training-free baseline: per image, a 2-component GMM on signed
           perpendicular distance; the minority component is IHC (~1:3
           prior). Zero training step, fully deterministic — the design
           note's headline advantage. Also the weird-region oracle.

  cv       GroupKFold-by-image CV of the learned model (logreg | gbm) on
           the geom table. Same protocol & metrics as the CNN, so numbers
           are directly comparable to ihc_ohc.md's table.

  train    one image-level split → pickled model (+ scaler, feature names)
           + held-out test report + coefficient / importance dump.

  fuse     the headline step. Per cell, combine the CNN's P(IHC) (read
           from the seg dict) with the geom model's P(IHC). Geom probs are
           generated *out-of-fold* (geom model refit per fold) so the
           fusion estimate is honest. Reports CNN-alone vs geom-alone vs
           fused, and a McNemar test of fused-vs-CNN so a gain is shown to
           be real, not noise.

  predict  one seg.npy → write class_map_geom / class_prob_geom /
           geom_flag (and, with --fuse + a fusion ckpt, class_map_fused /
           class_prob_fused) back in, non-destructively (masks untouched —
           plot_boxes.py and the GUI keep working).

The small metric/split/config helpers are deliberately re-implemented (not
imported from ihc_ohc_classifier) to keep this stack torch-free; that
module remains their canonical version — keep them in sync.

Dependencies (beyond the cellpose env): scikit-learn (already required for
the CNN's GroupKFold), scipy, scikit-image. No torch.

Usage (inside the cellpose container)
-------------------------------------
  python /helpers/ihc_ohc_geom.py --data_dir .../traintest/train --out /helpers/geom_train.npz
  python /helpers/ihc_ohc_geom.py --data_dir .../traintest/test  --out /helpers/geom_test.npz

  python /helpers/ihc_ohc_geom_clf.py rule  --config /helpers/ihc_ohc_geom_config.yaml
  python /helpers/ihc_ohc_geom_clf.py cv    --config /helpers/ihc_ohc_geom_config.yaml
  python /helpers/ihc_ohc_geom_clf.py train --config /helpers/ihc_ohc_geom_config.yaml
  # CNN probs must already be written into the segs:
  #   ihc_ohc_classifier.py predict --ckpt best.pt --seg <...>_seg.npy --write
  python /helpers/ihc_ohc_geom_clf.py fuse  --config /helpers/ihc_ohc_geom_config.yaml
  python /helpers/ihc_ohc_geom_clf.py predict --geom_ckpt /helpers/ihc_ohc_geom_run/geom_best.pkl \
      --seg .../000_..._seg.npy --fuse --fuse_ckpt /helpers/ihc_ohc_geom_run/fuse.pkl --write
"""

import argparse
import copy
import itertools
import json
import os
import pickle

import numpy as np
import yaml
from scipy.stats import chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ihc_ohc_crops import CLASS_NAMES, iter_seg_files, resolve_class_map, seg_stem
from ihc_ohc_geom import (
    FEATURE_NAMES, geom_features_for_seg, load_geom_npz,
)

IHC, OHC = 0, 1  # class indices, fixed (matches ihc_ohc_crops.CLASS_NAMES)


# ── config (same pattern as ihc_ohc_classifier; torch-free copy) ────────────────

DEFAULT_CONFIG = {
    "data": {
        "geom_train_npz": "/helpers/geom_train.npz",
        "geom_test_npz": "/helpers/geom_test.npz",
        # seg dirs are only needed by `fuse` (it reads the CNN's class_prob
        # straight from the seg files, aligned per cell id).
        "seg_train_dir": "/data/to_zip/hcat-data/Confocal/Cunningham/traintest/train",
        "seg_test_dir": "/data/to_zip/hcat-data/Confocal/Cunningham/traintest/test",
        "out_dir": "/helpers/ihc_ohc_geom_run",
    },
    "model": {
        "type": "logreg",          # logreg | gbm
        "class_weight": "balanced",  # mirror the CNN's bal_acc-first stance
        "C": 1.0,                  # logreg inverse-reg
        "n_estimators": 300,       # gbm
        "max_depth": 3,            # gbm
        "learning_rate": 0.05,     # gbm
        "k_neighbors": 6,          # must match the geom build
        "seed": 0,
    },
    "fuse": {
        "method": "mean",          # mean (1 weight, CV-picked) | stack (logreg)
        "weight_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                        0.6, 0.7, 0.8, 0.9, 1.0],  # w = CNN share
        "geom_source": "model",    # model (learned, OOF) | rule (training-free)
    },
    "cv": {"folds": 5, "val_frac": 0.2},
    "sweep": {},  # any model.* key → list of candidates, CV-scored
}


def _deep_merge(base, override, path=""):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        where = f"{path}{k}"
        if k not in out:
            raise KeyError(f"unknown config key '{where}'")
        if isinstance(out[k], dict) and isinstance(v, dict) and k != "sweep":
            out[k] = _deep_merge(out[k], v, path=f"{where}.")
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path):
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    cfg = _deep_merge(DEFAULT_CONFIG, user)
    for k in cfg.get("sweep", {}):
        if k not in DEFAULT_CONFIG["model"]:
            raise KeyError(f"sweep key '{k}' is not a model.* hyperparameter")
    return cfg


def print_config(cfg, tag="config"):
    print(f"=== {tag} ===")
    print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False).rstrip())
    print("=" * (len(tag) + 8))


# ── metrics & split (canonical copy lives in ihc_ohc_classifier.py) ─────────────

def metrics_from_cm(cm):
    acc = cm.trace() / max(cm.sum(), 1)
    recalls, f1s = [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        rec = tp / max(cm[i].sum(), 1)
        prec = tp / max(cm[:, i].sum(), 1)
        recalls.append(rec)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return dict(acc=acc, bal_acc=float(np.mean(recalls)),
                macro_f1=float(np.mean(f1s)), ihc_rec=recalls[IHC],
                ohc_rec=recalls[OHC])


def cm_from(y, p, n=2):
    cm = np.zeros((n, n), np.int64)
    for t, q in zip(y, p):
        cm[t, q] += 1
    return cm


def fmt_cm(cm):
    m = metrics_from_cm(cm)
    lines = ["          pred:" + "".join(f"{c:>7}" for c in CLASS_NAMES)]
    for i, c in enumerate(CLASS_NAMES):
        row = "".join(f"{v:>7}" for v in cm[i])
        rec = cm[i, i] / max(cm[i].sum(), 1)
        lines.append(f"  true {c:<5}{row}   recall={rec:.3f}")
    lines.append(f"  acc={m['acc']:.3f}  bal_acc={m['bal_acc']:.3f}  "
                 f"macro_f1={m['macro_f1']:.3f}")
    return "\n".join(lines)


def group_split(groups, val_frac, seed):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_g = set(uniq[:n_val].tolist())
    is_val = np.array([g in val_g for g in groups])
    return ~is_val, is_val


def mcnemar(y, pa, pb):
    """McNemar (continuity-corrected) on two classifiers' predictions.

    Tests whether classifier B (fused) differs from A (CNN) on the cells
    where exactly one is right. Returns (b, c, chi2, p): b = A-right /
    B-wrong, c = A-wrong / B-right; p small ⇒ the difference is real.
    """
    a_ok, b_ok = pa == y, pb == y
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return b, c, 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(stat), float(chi2.sf(stat, 1))


# ── learned geom model ──────────────────────────────────────────────────────────

def make_model(mcfg):
    """logreg or gbm. Both default to class-balanced weighting so the
    minority IHC is not sacrificed (the CNN's hard-won lesson)."""
    if mcfg["type"] == "gbm":
        # GBM has no class_weight; the caller passes balanced sample_weight.
        return GradientBoostingClassifier(
            n_estimators=mcfg["n_estimators"], max_depth=mcfg["max_depth"],
            learning_rate=mcfg["learning_rate"], random_state=mcfg["seed"])
    return LogisticRegression(
        C=mcfg["C"], class_weight=mcfg["class_weight"], max_iter=2000,
        random_state=mcfg["seed"])


def _sample_weight(y, mcfg):
    if mcfg["type"] != "gbm" or mcfg["class_weight"] != "balanced":
        return None
    cnt = np.bincount(y, minlength=2).astype(float)
    w = cnt.sum() / (2 * np.maximum(cnt, 1))
    return w[y]


def fit_geom(X, y, mcfg):
    """Standardise (fit on this X only) → fit the model. Returns
    (scaler, clf) reusable by predict_proba_ihc."""
    sc = StandardScaler().fit(X)
    clf = make_model(mcfg)
    clf.fit(sc.transform(X), y, sample_weight=_sample_weight(y, mcfg))
    return sc, clf


def proba_ihc(model, X):
    sc, clf = model
    return clf.predict_proba(sc.transform(X))[:, IHC]


# ── training-free rule ──────────────────────────────────────────────────────────

def rule_proba_ihc(feats, groups, *, ihc_prior=0.25):
    """Per image: 2-component GMM on signed perpendicular distance; the
    minority / off-centre component is IHC.

    The axis spline is fit through *all* centroids, so the OHC band (≈3:1)
    sits near perp≈0 and the single IHC row is offset to one side. We pick
    the GMM component with the smaller mixing weight as IHC; ties broken by
    larger |mean| (farther from the OHC-dominated axis). Returns P(IHC) per
    cell — no fit on labels anywhere, fully deterministic per image.
    """
    perp = feats[:, FEATURE_NAMES.index("perp_signed")]
    out = np.zeros(len(perp))
    for g in np.unique(groups):
        m = groups == g
        x = perp[m].reshape(-1, 1)
        if len(x) < 4 or np.allclose(x, x[0]):
            out[m] = ihc_prior          # too few / degenerate → prior
            continue
        gm = GaussianMixture(2, covariance_type="full", random_state=0,
                             n_init=3).fit(x)
        # IHC = minority component; tie-break on distance from axis.
        order = sorted(range(2), key=lambda k: (gm.weights_[k],
                                                 -abs(gm.means_[k, 0])))
        ihc_k = order[0]
        out[m] = gm.predict_proba(x)[:, ihc_k]
    return out


# ── per-seg assembly (for fuse / predict: aligns geom with the CNN) ──────────────

def _cnn_pihc(seg, cid):
    """P(IHC) for one cell from the CNN keys the classifier wrote.

    `class_prob` is the prob of the *predicted* class; invert when the
    prediction was OHC. None if the CNN never scored this cell.
    """
    pred = seg.get("class_map_pred", {})
    prob = seg.get("class_prob", {})
    if cid not in pred or cid not in prob:
        return None
    p = float(prob[cid])
    return p if pred[cid] == "IHC" else 1.0 - p


def assemble_segs(seg_dir, *, k_neighbors, need_cnn, need_gt,
                  include_augmented=False):
    """Walk seg_dir → aligned per-cell arrays.

    Returns dict with feats, p_cnn (P(IHC) or nan), y (or -1), groups,
    flags, and a (image_name, cell_id) ref per row. `need_cnn` keeps only
    cells the CNN scored; `need_gt` only labelled cells (eval). Used by
    fuse (need both) and predict (neither — score every cell).
    """
    F, PC, Y, G, FL, REF = [], [], [], [], [], []
    gi = {}
    n_missing_cnn = 0
    for seg_path, gkey in iter_seg_files(seg_dir, include_augmented):
        seg = np.load(seg_path, allow_pickle=True).item()
        if seg.get("masks") is None:
            continue
        gt = {}
        if need_gt:
            gt, _ = resolve_class_map(seg, seg_path, seg_dir)
            if not gt:
                continue
        cids, feats, flags, _ = geom_features_for_seg(
            seg, k_neighbors=k_neighbors)
        if not cids:
            continue
        name = seg_stem(seg_path)
        if gkey not in gi:
            gi[gkey] = len(gi)
        g = gi[gkey]
        for cid, f, fl in zip(cids, feats, flags):
            if need_gt and cid not in gt:
                continue
            pc = _cnn_pihc(seg, cid)
            if need_cnn and pc is None:
                n_missing_cnn += 1
                continue
            F.append(f)
            PC.append(np.nan if pc is None else pc)
            Y.append({"IHC": IHC, "OHC": OHC}[gt[cid]] if cid in gt else -1)
            G.append(g)
            FL.append(int(fl))
            REF.append((name, int(cid)))
    if not F:
        raise RuntimeError(
            f"no usable cells in {seg_dir}"
            + (f" ({n_missing_cnn} missing CNN class_prob — run the CNN "
               "`predict --write` first)" if need_cnn and n_missing_cnn else ""))
    if need_cnn and n_missing_cnn:
        print(f"  note: {n_missing_cnn} cells lacked CNN class_prob (skipped)")
    return dict(feats=np.stack(F).astype(np.float32),
                p_cnn=np.asarray(PC, float), y=np.asarray(Y, np.int64),
                groups=np.asarray(G, np.int64), flags=np.asarray(FL, np.int64),
                ref=REF)


# ── cv / train (learned geom model alone) ───────────────────────────────────────

def _splits(groups, cfg, seed):
    folds = int(cfg["cv"]["folds"])
    if folds <= 1:
        tr, va = group_split(groups, cfg["cv"]["val_frac"], seed)
        return [(np.where(tr)[0], np.where(va)[0])]
    return list(GroupKFold(n_splits=folds).split(np.zeros(len(groups)),
                                                 groups=groups))


def oof_geom(feats, y, groups, mcfg, splits):
    """Out-of-fold P(IHC) from the learned geom model — leak-free, so it
    can feed both the geom-alone report and the fusion stacker."""
    p = np.full(len(y), np.nan)
    for tr, va in splits:
        model = fit_geom(feats[tr], y[tr], mcfg)
        p[va] = proba_ihc(model, feats[va])
    return p


def cross_validate(cfg, *, verbose=True):
    d = load_geom_npz(cfg["data"]["geom_train_npz"])
    mcfg = cfg["model"]
    splits = _splits(d["groups"], cfg, mcfg["seed"])
    rows = []
    for k, (tr, va) in enumerate(splits, 1):
        model = fit_geom(d["feats"][tr], d["labels"][tr], mcfg)
        p = proba_ihc(model, d["feats"][va])
        pred = np.where(p >= 0.5, IHC, OHC)
        m = metrics_from_cm(cm_from(d["labels"][va], pred))
        rows.append(m)
        if verbose:
            ng = len(np.unique(d["groups"][va]))
            print(f"  fold {k}/{len(splits)}  val_imgs {ng:2d}  "
                  f"acc {m['acc']:.4f}  bal_acc {m['bal_acc']:.4f}  "
                  f"f1 {m['macro_f1']:.4f}")

    def ms(key):
        v = np.array([r[key] for r in rows])
        return {"mean": float(v.mean()), "std": float(v.std())}

    agg = {k: ms(k) for k in ("acc", "bal_acc", "macro_f1", "ihc_rec",
                              "ohc_rec")}
    if verbose:
        print(f"CV {len(splits)}-fold  "
              f"bal_acc {agg['bal_acc']['mean']:.4f} ± {agg['bal_acc']['std']:.4f}"
              f"  acc {agg['acc']['mean']:.4f} ± {agg['acc']['std']:.4f}  "
              f"macro_f1 {agg['macro_f1']['mean']:.4f}")
    return dict(folds=rows, mean_std=agg)


def train(cfg):
    d = load_geom_npz(cfg["data"]["geom_train_npz"])
    mcfg = cfg["model"]
    out_dir = cfg["data"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tr, va = group_split(d["groups"], cfg["cv"]["val_frac"], mcfg["seed"])
    model = fit_geom(d["feats"][tr], d["labels"][tr], mcfg)
    pv = proba_ihc(model, d["feats"][va])
    mv = metrics_from_cm(cm_from(d["labels"][va],
                                 np.where(pv >= 0.5, IHC, OHC)))
    print(f"val  acc {mv['acc']:.4f}  bal_acc {mv['bal_acc']:.4f}  "
          f"IHC_rec {mv['ihc_rec']:.4f}  OHC_rec {mv['ohc_rec']:.4f}")

    # refit on all train for the shipped model.
    final = fit_geom(d["feats"], d["labels"], mcfg)
    ckpt = dict(scaler=final[0], clf=final[1], model_cfg=mcfg,
                feature_names=d["feature_names"], class_names=CLASS_NAMES)
    ckpt_path = os.path.join(out_dir, "geom_best.pkl")
    with open(ckpt_path, "wb") as fh:
        pickle.dump(ckpt, fh)
    print(f"saved model → {ckpt_path}")
    _dump_importance(final, d["feature_names"])

    if cfg["data"]["geom_test_npz"]:
        t = load_geom_npz(cfg["data"]["geom_test_npz"])
        pt = proba_ihc(final, t["feats"])
        cm = cm_from(t["labels"], np.where(pt >= 0.5, IHC, OHC))
        print(f"\nTEST (geom alone)\n{fmt_cm(cm)}")
        with open(os.path.join(out_dir, "test_report.json"), "w") as fh:
            json.dump(metrics_from_cm(cm), fh, indent=2)


def _dump_importance(model, names):
    sc, clf = model
    if hasattr(clf, "coef_"):
        imp = clf.coef_[0]
        title = "logreg coef (standardised; + → IHC)"
    elif hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        title = "gbm feature importance"
    else:
        return
    order = np.argsort(-np.abs(imp))
    print(f"\n{title}:")
    for i in order[:12]:
        print(f"  {names[i]:>16s}  {imp[i]:+.3f}")


# ── sweep ───────────────────────────────────────────────────────────────────────

def sweep(cfg):
    sw = cfg.get("sweep", {})
    grid = ([{}] if not sw else
            [dict(zip(sw, c)) for c in itertools.product(
                *[v if isinstance(v, list) else [v] for v in sw.values()])])
    out_dir = cfg["data"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"sweep: {len(grid)} configs × {cfg['cv']['folds']}-fold CV")
    res = []
    for i, ov in enumerate(grid, 1):
        trial = copy.deepcopy(cfg)
        trial["model"].update(ov)
        tag = ", ".join(f"{k}={v}" for k, v in ov.items()) or "baseline"
        print(f"\n[{i}/{len(grid)}] {tag}")
        agg = cross_validate(trial, verbose=True)["mean_std"]
        res.append(dict(override=ov, bal_acc=agg["bal_acc"]["mean"],
                        bal_acc_std=agg["bal_acc"]["std"],
                        macro_f1=agg["macro_f1"]["mean"]))
    res.sort(key=lambda r: r["bal_acc"], reverse=True)
    print("\n=== sweep leaderboard (mean CV bal_acc) ===")
    for rk, r in enumerate(res, 1):
        tag = ", ".join(f"{k}={v}" for k, v in r["override"].items()) or "baseline"
        print(f"  {rk:2d}. bal_acc {r['bal_acc']:.4f} ± {r['bal_acc_std']:.4f}"
              f"  f1 {r['macro_f1']:.4f}  | {tag}")
    with open(os.path.join(out_dir, "sweep_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)


# ── fuse (the headline) ─────────────────────────────────────────────────────────

def _fuse_mean(p_cnn, p_geom, w):
    return w * p_cnn + (1.0 - w) * p_geom


def fuse(cfg):
    """CNN ⊕ geom late fusion, honestly estimated.

    Geom probs are out-of-fold (model refit per fold, or the training-free
    rule). The fusion weight (mean) / stacker (logreg) is picked on the
    same OOF predictions, then frozen and applied to the held-out test set.
    Reports CNN-alone vs geom-alone vs fused + a McNemar of fused-vs-CNN.
    """
    mcfg, fcfg = cfg["model"], cfg["fuse"]
    out_dir = cfg["data"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    kN = mcfg["k_neighbors"]

    print(f"assembling train segs ({cfg['data']['seg_train_dir']}) …")
    tr = assemble_segs(cfg["data"]["seg_train_dir"], k_neighbors=kN,
                       need_cnn=True, need_gt=True)
    print(f"  {len(tr['y'])} cells, {len(np.unique(tr['groups']))} images")
    splits = _splits(tr["groups"], cfg, mcfg["seed"])

    # out-of-fold geom P(IHC)
    if fcfg["geom_source"] == "rule":
        pg = rule_proba_ihc(tr["feats"], tr["groups"])
    else:
        pg = oof_geom(tr["feats"], tr["y"], tr["groups"], mcfg, splits)
    pc, y = tr["p_cnn"], tr["y"]

    def report(tag, p):
        cm = cm_from(y, np.where(p >= 0.5, IHC, OHC))
        m = metrics_from_cm(cm)
        print(f"\n[{tag}]  acc {m['acc']:.4f}  bal_acc {m['bal_acc']:.4f}  "
              f"IHC_rec {m['ihc_rec']:.4f}  OHC_rec {m['ohc_rec']:.4f}  "
              f"f1 {m['macro_f1']:.4f}")
        return cm, m

    _, m_cnn = report("CNN alone (OOF train)", pc)
    report("geom alone (OOF train)", pg)

    fused_cfg = {}
    if fcfg["method"] == "stack":
        # logreg on [p_cnn, p_geom]; OOF-stacked to avoid optimism.
        Z = np.column_stack([pc, pg])
        p_fused = np.full(len(y), np.nan)
        for trn, val in splits:
            st = LogisticRegression(class_weight=mcfg["class_weight"],
                                    max_iter=2000)
            st.fit(Z[trn], y[trn])
            p_fused[val] = st.predict_proba(Z[val])[:, IHC]
        st_full = LogisticRegression(class_weight=mcfg["class_weight"],
                                     max_iter=2000).fit(Z, y)
        fused_cfg = dict(method="stack",
                         coef=st_full.coef_[0].tolist(),
                         intercept=float(st_full.intercept_[0]))
    else:  # mean — pick w on the OOF grid (1 dof, coarse grid → low risk)
        best_w, best_b = 1.0, -1.0
        for w in fcfg["weight_grid"]:
            b = metrics_from_cm(cm_from(
                y, np.where(_fuse_mean(pc, pg, w) >= 0.5, IHC, OHC)))["bal_acc"]
            if b > best_b:
                best_b, best_w = b, w
        p_fused = _fuse_mean(pc, pg, best_w)
        fused_cfg = dict(method="mean", weight=best_w)
        print(f"\nchosen CNN weight w={best_w:.2f} "
              f"(1-w on geom), OOF bal_acc {best_b:.4f}")

    _, m_fused = report(f"FUSED ({fcfg['method']}, OOF train)", p_fused)
    b, c, st, pval = mcnemar(y, np.where(pc >= 0.5, IHC, OHC),
                             np.where(p_fused >= 0.5, IHC, OHC))
    print(f"\nMcNemar fused-vs-CNN: CNN-only-right={b}  fused-only-right={c}  "
          f"χ²={st:.3f}  p={pval:.4g}  "
          f"→ {'significant' if pval < 0.05 else 'not significant'} "
          f"({'gain' if c > b else 'no gain'})")

    ckpt = dict(fuse=fused_cfg, geom_source=fcfg["geom_source"],
                model_cfg=mcfg, k_neighbors=kN,
                cnn_oof=m_cnn, fused_oof=m_fused, mcnemar=dict(
                    b=b, c=c, chi2=st, p=pval))
    ckpt_path = os.path.join(out_dir, "fuse.pkl")
    with open(ckpt_path, "wb") as fh:
        pickle.dump(ckpt, fh)
    print(f"saved fusion ckpt → {ckpt_path}")

    # held-out test: refit geom on ALL train, freeze fusion, score test segs.
    if cfg["data"].get("seg_test_dir"):
        print(f"\nassembling test segs ({cfg['data']['seg_test_dir']}) …")
        te = assemble_segs(cfg["data"]["seg_test_dir"], k_neighbors=kN,
                           need_cnn=True, need_gt=True)
        if fcfg["geom_source"] == "rule":
            pg_t = rule_proba_ihc(te["feats"], te["groups"])
        else:
            gm = fit_geom(tr["feats"], tr["y"], mcfg)
            pg_t = proba_ihc(gm, te["feats"])
        pc_t, y_t = te["p_cnn"], te["y"]
        if fused_cfg["method"] == "stack":
            coef = np.array(fused_cfg["coef"])
            z = np.column_stack([pc_t, pg_t]) @ coef + fused_cfg["intercept"]
            pf_t = 1.0 / (1.0 + np.exp(-z))
        else:
            pf_t = _fuse_mean(pc_t, pg_t, fused_cfg["weight"])
        print(f"\n=== HELD-OUT TEST ({len(y_t)} cells) ===")
        for tag, p in (("CNN alone", pc_t), ("geom alone", pg_t),
                       ("FUSED", pf_t)):
            cm = cm_from(y_t, np.where(p >= 0.5, IHC, OHC))
            print(f"\n[{tag}]\n{fmt_cm(cm)}")
        bt, ct, stt, pt = mcnemar(y_t, np.where(pc_t >= 0.5, IHC, OHC),
                                  np.where(pf_t >= 0.5, IHC, OHC))
        print(f"\nMcNemar (test) fused-vs-CNN: CNN-only-right={bt}  "
              f"fused-only-right={ct}  χ²={stt:.3f}  p={pt:.4g}")


# ── predict (write back into one seg) ───────────────────────────────────────────

def predict_seg_geom(geom_ckpt, seg_path, *, fuse_ckpt=None, write=False):
    """Score every instance in one seg.npy with the geom model (and,
    given a fusion ckpt + the CNN keys, the fused decision). Writes
    class_map_geom / class_prob_geom / geom_flag (+ _fused) — masks and
    the CNN's keys are left untouched."""
    with open(geom_ckpt, "rb") as fh:
        gk = pickle.load(fh)
    model = (gk["scaler"], gk["clf"])
    kN = gk["model_cfg"]["k_neighbors"]

    seg = np.load(seg_path, allow_pickle=True).item()
    cids, feats, flags, _ = geom_features_for_seg(seg, k_neighbors=kN)
    if not cids:
        print("no instances")
        return {}
    pg = proba_ihc(model, feats)
    geom_map = {int(c): (CLASS_NAMES[IHC] if p >= 0.5 else CLASS_NAMES[OHC])
                for c, p in zip(cids, pg)}
    geom_prob = {int(c): float(max(p, 1 - p)) for c, p in zip(cids, pg)}
    flag_map = {int(c): int(f) for c, f in zip(cids, flags)}

    fused_map = None
    if fuse_ckpt:
        with open(fuse_ckpt, "rb") as fh:
            fk = pickle.load(fh)
        pc = np.array([_cnn_pihc(seg, int(c)) for c in cids], float)
        if np.isnan(pc).any():
            print("  fusion skipped: CNN class_prob missing for some cells "
                  "(run ihc_ohc_classifier.py predict --write first)")
        else:
            f = fk["fuse"]
            if f["method"] == "stack":
                z = (np.column_stack([pc, pg]) @ np.array(f["coef"])
                     + f["intercept"])
                pf = 1.0 / (1.0 + np.exp(-z))
            else:
                pf = f["weight"] * pc + (1 - f["weight"]) * pg
            fused_map = {int(c): (CLASS_NAMES[IHC] if p >= 0.5
                                  else CLASS_NAMES[OHC])
                         for c, p in zip(cids, pf)}

    n_ihc = sum(v == "IHC" for v in geom_map.values())
    print(f"{os.path.basename(seg_path)}: {len(cids)} cells  "
          f"geom IHC {n_ihc}/OHC {len(cids)-n_ihc}  "
          f"flagged {sum(v>0 for v in flag_map.values())}")
    gt, src = resolve_class_map(seg, seg_path, os.path.dirname(seg_path))
    if gt:
        pairs = [("geom", geom_map)]
        if fused_map:
            pairs.append(("fused", fused_map))
        for tag, mp in pairs:
            ok = sum(gt.get(c) == cl for c, cl in mp.items() if c in gt)
            tot = sum(c in gt for c in mp)
            if tot:
                print(f"  {tag} vs GT [{src}]: acc {ok/tot:.4f} ({ok}/{tot})")

    if write:
        seg["class_map_geom"] = geom_map
        seg["class_prob_geom"] = geom_prob
        seg["geom_flag"] = flag_map
        if fused_map:
            seg["class_map_fused"] = fused_map
        np.save(seg_path, seg)
        print(f"  wrote class_map_geom / class_prob_geom / geom_flag"
              + (" / class_map_fused" if fused_map else "")
              + f" → {os.path.basename(seg_path)}")
    return geom_map


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="IHC/OHC geometric classifier + CNN fusion")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, h in (("rule", "training-free GMM baseline"),
                    ("cv", "GroupKFold CV of the learned geom model"),
                    ("train", "one split → pickled model + test report"),
                    ("sweep", "CV-score the sweep grid"),
                    ("fuse", "CNN ⊕ geom late fusion + McNemar")):
        s = sub.add_parser(name, help=h)
        s.add_argument("--config", required=True)
    q = sub.add_parser("predict", help="score one seg.npy, write geom/fused back")
    q.add_argument("--geom_ckpt", required=True)
    q.add_argument("--seg", required=True)
    q.add_argument("--fuse_ckpt", default=None)
    q.add_argument("--fuse", action="store_true",
                   help="also write the fused decision (needs --fuse_ckpt + CNN keys)")
    q.add_argument("--write", action="store_true")
    return p.parse_args()


def rule_cmd(cfg):
    """Evaluate the training-free rule on the train table (and test, if set)."""
    d = load_geom_npz(cfg["data"]["geom_train_npz"])
    p = rule_proba_ihc(d["feats"], d["groups"])
    cm = cm_from(d["labels"], np.where(p >= 0.5, IHC, OHC))
    print(f"RULE (train, no fit)\n{fmt_cm(cm)}")
    if cfg["data"].get("geom_test_npz"):
        t = load_geom_npz(cfg["data"]["geom_test_npz"])
        pt = rule_proba_ihc(t["feats"], t["groups"])
        print(f"\nRULE (test, no fit)\n"
              f"{fmt_cm(cm_from(t['labels'], np.where(pt>=0.5, IHC, OHC)))}")


def main():
    args = parse_args()
    if args.cmd == "predict":
        predict_seg_geom(args.geom_ckpt, args.seg,
                         fuse_ckpt=args.fuse_ckpt if args.fuse else None,
                         write=args.write)
        return
    cfg = load_config(args.config)
    print_config(cfg, tag=f"{args.cmd} config ({args.config})")
    {"rule": rule_cmd, "cv": cross_validate, "train": train,
     "sweep": sweep, "fuse": fuse}[args.cmd](cfg)


if __name__ == "__main__":
    main()
