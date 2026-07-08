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

Every `train` / `sweep` / `fuse` writes to a fresh timestamped subdir
under `data.out_dir` — e.g. runs/20260520-104530_train-geom/. The resolved
config is frozen alongside as `config.yaml`. `--run_dir <path>` overrides
the auto-stamp (the pipeline orchestrator uses this to keep all steps of
one pipeline run under one timestamp).

Usage (inside the cellpose container)
-------------------------------------
  python /helpers/ihc_ohc/ihc_ohc_geom.py --data_dir .../traintest/train \
      --out /helpers/ihc_ohc/runs/cache/geom_train.npz
  python /helpers/ihc_ohc/ihc_ohc_geom.py --data_dir .../traintest/test  \
      --out /helpers/ihc_ohc/runs/cache/geom_test.npz

  python /helpers/ihc_ohc/ihc_ohc_geom_clf.py rule  --config /helpers/ihc_ohc/configs/geom.yaml
  python /helpers/ihc_ohc/ihc_ohc_geom_clf.py cv    --config /helpers/ihc_ohc/configs/geom.yaml
  python /helpers/ihc_ohc/ihc_ohc_geom_clf.py train --config /helpers/ihc_ohc/configs/geom.yaml
  # CNN probs must already be written into the segs:
  #   ihc_ohc_classifier.py predict --ckpt best.pt --seg <...>_seg.npy --write
  python /helpers/ihc_ohc/ihc_ohc_geom_clf.py fuse  --config /helpers/ihc_ohc/configs/geom.yaml
  python /helpers/ihc_ohc/ihc_ohc_geom_clf.py predict \
      --geom_ckpt /helpers/ihc_ohc/runs/<stamp>_train-geom/geom_best.pkl \
      --seg .../000_..._seg.npy --fuse \
      --fuse_ckpt /helpers/ihc_ohc/runs/<stamp>_fuse/fuse.pkl --write
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ihc_ohc_crops import (
    CLASS_NAMES, active_reject_ids, extract_cell_crop, iter_seg_files,
    load_image_plane, load_pred, new_run_dir, resolve_class_map,
    resolve_training_label_map, save_run_config, seg_stem, tif_for_seg,
    update_pred,
)
from ihc_ohc_geom import (
    FEATURE_NAMES, geom_features_for_seg, load_geom_npz,
)

IHC, OHC = 0, 1  # class indices, fixed (matches ihc_ohc_crops.CLASS_NAMES)


# ── config (same pattern as ihc_ohc_classifier; torch-free copy) ────────────────

DEFAULT_CONFIG = {
    "data": {
        "geom_train_npz": "/helpers/ihc_ohc/runs/cache/geom_train.npz",
        "geom_test_npz": "/helpers/ihc_ohc/runs/cache/geom_test.npz",
        # `fuse` works at the seg-file level (aligns geom & CNN per cell id).
        "seg_train_dir": "/data/to_zip/hcat-data/Confocal/Cunningham/traintest/train",
        "seg_test_dir": "/data/to_zip/hcat-data/Confocal/Cunningham/traintest/test",
        # the CNN side of fusion: its train config (for the OOF retrain) and
        # the crops .npz (only its `meta`, for matching crop geometry).
        "cnn_config": "/helpers/ihc_ohc/configs/cnn.yaml",
        "crops_train_npz": "/helpers/ihc_ohc/runs/cache/crops_train.npz",
        # leak-free group key for the fuse-side seg walk — must match the
        # builders' --group_mode: "numeric" (Cunningham) | "clc" (animal id).
        "group_mode": "numeric",
        # Root for run artifacts; each train/sweep/fuse gets a fresh
        # timestamped subdir (e.g., runs/20260520-104530_train-geom/).
        "out_dir": "/helpers/ihc_ohc/runs",
        # Deployed checkpoints — read by ihc_ohc_pipeline.py predict (this
        # script doesn't use them; declared here so the merge accepts the
        # fields). Update after each train/fuse run; null → caller must
        # supply --geom_ckpt / --fuse_ckpt.
        "geom_ckpt": None,
        "fuse_ckpt": None,
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
        "cnn_oof": True,           # True  → retrain the CNN per fold for
                                   #   leak-free train-set probs (correct;
                                   #   the weight/stacker is then chosen on
                                   #   an honest signal). Slow: one CNN per
                                   #   fold. False → read the deployed CNN's
                                   #   class_prob from the train segs
                                   #   (in-sample, biased — diagnostics only).
        "threshold_grid": [round(0.30 + i * 0.02, 3) for i in range(21)],
                                   # candidate decision thresholds for the
                                   # *fused* score (0.30 … 0.70). Picked
                                   # jointly with `w` on OOF for bal_acc;
                                   # 0.5 is no longer optimal once
                                   # probabilities are calibrated against
                                   # the real ~1:3 prior. Component-alone
                                   # rows stay at 0.5 for comparability.
        "calibrate": "isotonic",   # isotonic | platt | none. Per-component
                                   #   monotonic recalibration fit on OOF
                                   #   train predictions, frozen, applied to
                                   #   test. Both models train against a
                                   #   balanced prior (sampler / class_weight)
                                   #   so their probabilities are
                                   #   balanced-prior calibrated, not
                                   #   prevalence calibrated — fix that here
                                   #   so `mean` fusion is on equal scales
                                   #   and downstream confidences are honest.
        "calibrate_components": ["cnn", "geom"],  # subset of {cnn, geom}
        # Per-image row-consistency post-pass applied at *inference* time
        # (predict_seg_geom), after fusion. IHC/OHC separate almost
        # perfectly by signed perp offset; this re-anchors the two perp
        # bands on the confident fused calls and flips any *low-confidence*
        # fused label whose perp position clearly belongs to the other
        # band. Confidence-gated → never overrides a confident call. The
        # params ride inside fuse.pkl so deployed/GUI inference applies the
        # same rule. Off by default (Cunningham unaffected); CLC enables it.
        "row_consistency": {
            "enabled": False,
            "conf_anchor": 0.85,    # fused conf ≥ this anchors the bands
            "conf_override": 0.60,  # only flip fused labels below this conf
            "k_anchor": 15,         # nearest perp anchors averaged per band
        },
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


# ── probability calibration (monotonic recalibrators for late fusion) ───────────

def ece(probs, y, n_bins=15):
    """Expected calibration error — mean |bin_mean_prob − bin_accuracy|,
    weighted by bin population. Lower is better; 0 = perfect calibration.

    `probs` is P(IHC); we evaluate calibration on the predicted-class
    confidence p* = max(p, 1−p) vs whether the argmax is correct, which is
    the standard top-label ECE used for binary classifiers.
    """
    probs = np.asarray(probs, float)
    y = np.asarray(y, int)
    pred = (probs >= 0.5).astype(int).clip(0, 1)
    pred = np.where(pred == 1, IHC, OHC)
    correct = (pred == y).astype(float)
    conf = np.maximum(probs, 1.0 - probs)
    edges = np.linspace(0.5, 1.0, n_bins + 1)            # conf ∈ [0.5, 1]
    edges[-1] += 1e-9
    err = 0.0
    n = len(y)
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.any():
            err += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return float(err)


def fit_calibrator(probs, y, method="isotonic"):
    """Fit a monotonic recalibrator P(IHC) → P(IHC | real prior).

    `isotonic`: IsotonicRegression on (probs → label==IHC). Flexible
    non-parametric fit; we have ~1500 IHC + ~4500 OHC OOF cells → plenty.
    `platt`: 1-parameter logistic (LogisticRegression on the single feature
    `probs`). For small-sample regimes — we don't need it but it's a
    sensible fallback. `none`: identity calibrator. Returns an object with
    a `.predict(probs)` method, so the caller never branches on method.
    """
    y = (np.asarray(y) == IHC).astype(int)
    p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6)
    if method == "none":
        class _Id:
            def predict(self, x):
                return np.clip(np.asarray(x, float), 0.0, 1.0)
        return _Id()
    if method == "platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(p.reshape(-1, 1), y)
        class _Platt:
            def __init__(self, m):
                self.m = m
            def predict(self, x):
                x = np.clip(np.asarray(x, float).reshape(-1, 1), 1e-6, 1 - 1e-6)
                return self.m.predict_proba(x)[:, 1]
        return _Platt(lr)
    # isotonic (default)
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                              y_max=1.0).fit(p, y)


def apply_calibrator(cal, probs):
    """Transform raw probs through the recalibrator. Single-line wrapper so
    callers don't need to know the calibrator's internal API."""
    return np.asarray(cal.predict(np.asarray(probs, float)), float)


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

def _crop_params(crops_npz):
    """CNN crop geometry from the crops .npz `meta` (the authoritative
    record of how the CNN was cropped). Only `meta` is touched — np.load
    is lazy, so the big `crops` array is never decompressed."""
    d = np.load(crops_npz, allow_pickle=True)
    m = d["meta"][0] if "meta" in d else {}
    m = m if isinstance(m, dict) else dict(m)
    return dict(out_size=int(m.get("out_size", 64)),
                pad_frac=float(m.get("pad_frac", 0.5)),
                pad_px=m.get("pad_px"),
                pad_value=m.get("pad_value", "mean"),
                soft_mask=bool(m.get("soft_mask", True)),
                channel=int(m.get("channel", 1)))


def _cnn_pihc(pred, cid):
    """P(IHC) for one cell from a loaded sidecar (or seg-fallback) dict.

    `class_prob` is the prob of the *predicted* class; invert when the
    prediction was OHC. None if the CNN never scored this cell. `pred`
    comes from `load_pred(seg_path, seg)` — that function transparently
    reads the sidecar when it exists and falls back to legacy in-seg keys.
    """
    pm = pred.get("class_map_pred", {}) or {}
    pp = pred.get("class_prob", {}) or {}
    if cid not in pm or cid not in pp:
        return None
    p = float(pp[cid])
    return p if pm[cid] == "IHC" else 1.0 - p


def assemble_segs(seg_dir, *, k_neighbors, need_cnn, need_gt,
                  with_crops=False, crop_params=None,
                  include_augmented=False, group_mode="numeric"):
    """Walk seg_dir → aligned per-cell arrays.

    Returns dict with feats, p_cnn (P(IHC) or nan), y (or -1), groups,
    flags, a (image_name, cell_id) ref per row, and — when `with_crops` —
    the CNN crop (2,S,S) per row, extracted with the *same* geometry the
    CNN was trained on (`crop_params`) so the OOF-CNN retrain sees inputs
    identical to deployment. `need_cnn` keeps only cells the deployed CNN
    scored; `need_gt` only labelled cells. Every kept array stays index-
    aligned (a cell is dropped from *all* of them or none).
    """
    F, PC, Y, G, FL, REF, CR = [], [], [], [], [], [], []
    gi = {}
    n_missing_cnn = 0
    cp = crop_params or {}
    for seg_path, gkey in iter_seg_files(seg_dir, include_augmented, group_mode):
        seg = np.load(seg_path, allow_pickle=True).item()
        masks = seg.get("masks")
        if masks is None:
            continue
        gt = {}
        if need_gt:
            # training-label resolver, not resolve_class_map: picks up the
            # GUI display labels (user > fused > geom > pred) for datasets
            # with no dataset-level GT (CLC). Identical to GT for Cunningham.
            gt, _ = resolve_training_label_map(seg, seg_path, seg_dir)
            if not gt:
                continue
        cids, feats, flags, _ = geom_features_for_seg(
            seg, k_neighbors=k_neighbors)
        if not cids:
            continue
        pred = load_pred(seg_path, seg) if need_cnn else {}
        plane = fill = None
        if with_crops:
            plane = load_image_plane(seg, tif_for_seg(seg_path),
                                     cp.get("channel", 1))
            pv = cp.get("pad_value", "mean")
            fill = float(plane.mean()) if pv == "mean" else float(pv)
        name = seg_stem(seg_path)
        if gkey not in gi:
            gi[gkey] = len(gi)
        g = gi[gkey]
        for cid, f, fl in zip(cids, feats, flags):
            if need_gt and cid not in gt:
                continue
            pc = _cnn_pihc(pred, cid) if pred else None
            if need_cnn and pc is None:
                n_missing_cnn += 1
                continue
            crop = None
            if with_crops:
                crop = extract_cell_crop(
                    plane, masks, int(cid), out_size=cp["out_size"],
                    pad_frac=cp["pad_frac"], pad_px=cp["pad_px"],
                    pad_value=fill, soft_mask=cp["soft_mask"])
                if crop is None:        # mask erased upstream → drop the row
                    continue
                CR.append(crop)
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
                ref=REF,
                crops=np.stack(CR).astype(np.float32) if with_crops else None)


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


def oof_cnn(crops, y, groups, splits, cfg):
    """Out-of-fold P(IHC) from the CNN — the *correctness* fix for fusion.

    The deployed CNN (best.pt) is trained on the whole train set, so its
    class_prob on training cells is in-sample: selecting the fusion
    weight/stacker against it over-trusts the CNN and the train-side
    rows are inflated. Here the CNN is retrained per outer fold (with a
    nested image-level split inside the fold-train for early stopping) and
    predicts only the held-out fold — exactly the leak-free protocol
    `oof_geom` uses, on the *same* `splits`, so the two OOF signals are
    jointly honest and aligned. The held-out *test* path is unaffected
    (it rightly uses the deployed best.pt — test was never seen).

    Slow by design: one CNN training per fold. torch + the classifier are
    imported lazily so `rule`/`cv`/`train`/`predict` stay torch-free.
    """
    import torch

    from ihc_ohc_classifier import TinyHCNet
    from ihc_ohc_classifier import fit as cnn_fit
    from ihc_ohc_classifier import load_config as cnn_load_config
    from ihc_ohc_classifier import predict_crops

    tcfg = cnn_load_config(cfg["data"]["cnn_config"])["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(splits, 1):
        # nest a held-out sub-val *inside* the fold-train (by image) for the
        # CNN's early stopping, so the fold-val stays fully untouched.
        es_tr_m, es_va_m = group_split(groups[tr], cfg["cv"]["val_frac"],
                                       tcfg["seed"])
        cnn_tr, cnn_es = tr[es_tr_m], tr[es_va_m]
        r = cnn_fit(crops, y, cnn_tr, cnn_es, tcfg, device, verbose=False)
        model = TinyHCNet(in_ch=crops.shape[1]).to(device)
        model.load_state_dict(r["best_state"])
        _, probs = predict_crops(model, crops[va], r["mean"], r["std"],
                                 device)
        p[va] = probs[:, IHC]
        print(f"  CNN OOF fold {k}/{len(splits)}: train {len(cnn_tr)} / "
              f"es {len(cnn_es)} cells → predicted {len(va)} "
              f"(CNN best ep {r['best_ep']})", flush=True)
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


def train(cfg, *, run_dir=None):
    """One image-level split → pickled model + held-out test report.

    Writes geom_best.pkl (+ test_report.json) into a fresh timestamped
    subdir under cfg.data.out_dir, plus a frozen config.yaml.
    """
    d = load_geom_npz(cfg["data"]["geom_train_npz"])
    mcfg = cfg["model"]
    run_dir = new_run_dir(cfg["data"]["out_dir"], "train-geom", run_dir)
    save_run_config(run_dir, cfg)
    print(f"run dir → {run_dir}")

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
    ckpt_path = os.path.join(run_dir, "geom_best.pkl")
    with open(ckpt_path, "wb") as fh:
        pickle.dump(ckpt, fh)
    print(f"saved model → {ckpt_path}")
    _dump_importance(final, d["feature_names"])

    if cfg["data"]["geom_test_npz"]:
        t = load_geom_npz(cfg["data"]["geom_test_npz"])
        pt = proba_ihc(final, t["feats"])
        cm = cm_from(t["labels"], np.where(pt >= 0.5, IHC, OHC))
        print(f"\nTEST (geom alone)\n{fmt_cm(cm)}")
        with open(os.path.join(run_dir, "test_report.json"), "w") as fh:
            json.dump(metrics_from_cm(cm), fh, indent=2)
    return run_dir


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

def sweep(cfg, *, run_dir=None):
    sw = cfg.get("sweep", {})
    grid = ([{}] if not sw else
            [dict(zip(sw, c)) for c in itertools.product(
                *[v if isinstance(v, list) else [v] for v in sw.values()])])
    run_dir = new_run_dir(cfg["data"]["out_dir"], "sweep-geom", run_dir)
    save_run_config(run_dir, cfg)
    print(f"run dir → {run_dir}")
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
    with open(os.path.join(run_dir, "sweep_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return run_dir


# ── fuse (the headline) ─────────────────────────────────────────────────────────

def _fuse_mean(p_cnn, p_geom, w):
    return w * p_cnn + (1.0 - w) * p_geom


def fuse(cfg, *, run_dir=None):
    """CNN ⊕ geom late fusion, honestly estimated.

    *Both* train-side signals are out-of-fold on the same `splits`: geom
    refit per fold (`oof_geom`), CNN retrained per fold (`oof_cnn`, when
    `fuse.cnn_oof`). The fusion weight (mean) / stacker (logreg) is picked
    on those jointly-honest OOF predictions, frozen, then applied to the
    held-out test set — where the *deployed* best.pt is rightly used (test
    was never seen). Reports CNN/geom/fused + a McNemar of fused-vs-CNN.

    Writes fuse.pkl into a fresh timestamped subdir under cfg.data.out_dir,
    plus a frozen config.yaml.
    """
    mcfg, fcfg = cfg["model"], cfg["fuse"]
    run_dir = new_run_dir(cfg["data"]["out_dir"], "fuse", run_dir)
    save_run_config(run_dir, cfg)
    print(f"run dir → {run_dir}")
    kN = mcfg["k_neighbors"]
    cnn_oof = bool(fcfg.get("cnn_oof", True))

    gm = cfg["data"].get("group_mode", "numeric")
    print(f"assembling train segs ({cfg['data']['seg_train_dir']}) …")
    if cnn_oof:
        cp = _crop_params(cfg["data"]["crops_train_npz"])
        tr = assemble_segs(cfg["data"]["seg_train_dir"], k_neighbors=kN,
                           need_cnn=False, need_gt=True,
                           with_crops=True, crop_params=cp, group_mode=gm)
    else:
        tr = assemble_segs(cfg["data"]["seg_train_dir"], k_neighbors=kN,
                           need_cnn=True, need_gt=True, group_mode=gm)
    print(f"  {len(tr['y'])} cells, {len(np.unique(tr['groups']))} images")
    splits = _splits(tr["groups"], cfg, mcfg["seed"])

    # out-of-fold geom P(IHC)
    if fcfg["geom_source"] == "rule":
        pg = rule_proba_ihc(tr["feats"], tr["groups"])
    else:
        pg = oof_geom(tr["feats"], tr["y"], tr["groups"], mcfg, splits)
    # out-of-fold (honest) or in-sample (biased, diagnostics-only) CNN P(IHC)
    if cnn_oof:
        print(f"computing OOF CNN — retraining {len(splits)} CNNs (the "
              f"slow, honest path) …", flush=True)
        pc = oof_cnn(tr["crops"], tr["y"], tr["groups"], splits, cfg)
    else:
        pc = tr["p_cnn"]
    y = tr["y"]
    tlbl = "OOF train" if cnn_oof else "in-sample train, BIASED"

    # Probability calibration: per-component monotonic recalibrators fit
    # on the OOF train predictions, then frozen. Both base models train
    # against a balanced prior, so their probabilities are balanced-prior
    # calibrated, not prevalence calibrated → fix before fusing on a
    # common scale. The transforms are monotonic, so argmax-accuracy is
    # essentially preserved (the wins are in ECE and `mean`-fusion sanity).
    cal_method = fcfg.get("calibrate", "isotonic")
    cal_set = set(fcfg.get("calibrate_components", ["cnn", "geom"]))
    pc_raw, pg_raw = pc.copy(), pg.copy()
    cal_cnn = fit_calibrator(pc_raw, y,
                             cal_method if "cnn" in cal_set else "none")
    cal_geom = fit_calibrator(pg_raw, y,
                              cal_method if "geom" in cal_set else "none")
    pc = apply_calibrator(cal_cnn, pc_raw)
    pg = apply_calibrator(cal_geom, pg_raw)
    print(f"\ncalibration: method={cal_method}  components={sorted(cal_set)}")
    print(f"  ECE  CNN  {ece(pc_raw, y):.4f} → {ece(pc, y):.4f}")
    print(f"  ECE  geom {ece(pg_raw, y):.4f} → {ece(pg, y):.4f}")

    def report(tag, p, th=0.5):
        cm = cm_from(y, np.where(p >= th, IHC, OHC))
        m = metrics_from_cm(cm)
        thtag = f" @th={th:.3f}" if th != 0.5 else ""
        print(f"\n[{tag}{thtag}]  acc {m['acc']:.4f}  bal_acc {m['bal_acc']:.4f}  "
              f"IHC_rec {m['ihc_rec']:.4f}  OHC_rec {m['ohc_rec']:.4f}  "
              f"f1 {m['macro_f1']:.4f}")
        return cm, m

    _, m_cnn = report(f"CNN alone ({tlbl})", pc)
    report("geom alone (OOF train)", pg)

    fused_cfg = {}
    th_grid = fcfg.get("threshold_grid", [0.5])
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
        # threshold tuning on OOF for bal_acc (1 hyperparameter)
        best_th, best_b = 0.5, -1.0
        for th in th_grid:
            b = metrics_from_cm(cm_from(
                y, np.where(p_fused >= th, IHC, OHC)))["bal_acc"]
            if b > best_b:
                best_b, best_th = b, th
        fused_cfg = dict(method="stack",
                         coef=st_full.coef_[0].tolist(),
                         intercept=float(st_full.intercept_[0]),
                         threshold=best_th)
        print(f"\nstack threshold={best_th:.3f}  OOF bal_acc {best_b:.4f}")
    else:  # mean — joint (w, threshold) search on OOF for bal_acc
        best_w, best_th, best_b = 0.5, 0.5, -1.0
        for w in fcfg["weight_grid"]:
            pf = _fuse_mean(pc, pg, w)
            for th in th_grid:
                b = metrics_from_cm(cm_from(
                    y, np.where(pf >= th, IHC, OHC)))["bal_acc"]
                if b > best_b:
                    best_b, best_w, best_th = b, w, th
        p_fused = _fuse_mean(pc, pg, best_w)
        fused_cfg = dict(method="mean", weight=best_w, threshold=best_th)
        print(f"\nchosen w={best_w:.2f} (CNN share)  threshold={best_th:.3f}  "
              f"OOF bal_acc {best_b:.4f}")
    best_th = fused_cfg["threshold"]

    _, m_fused = report(f"FUSED ({fcfg['method']}, {tlbl})", p_fused, best_th)
    print(f"  ECE  fused {ece(p_fused, y):.4f}")
    # Pairwise McNemar on the OOF predictions (component-alone calls at the
    # 0.5 argmax; fused at its tuned threshold). fused-vs-geom is the
    # decisive "does fusion beat the *better* single model" test — geom is
    # the stronger component here, so fused-vs-CNN overstates fusion's value.
    cnn_pred = np.where(pc >= 0.5, IHC, OHC)
    geom_pred = np.where(pg >= 0.5, IHC, OHC)
    fused_pred = np.where(p_fused >= best_th, IHC, OHC)

    def _mc(tag, a_lbl, predA, b_lbl, predB):
        ba, ca, sta, pa = mcnemar(y, predA, predB)
        verdict = ("not significant" if pa >= 0.05 else
                   f"significant ({'gain' if ca > ba else 'no gain'})")
        print(f"McNemar {tag}: {a_lbl}-only-right={ba}  {b_lbl}-only-right={ca}  "
              f"χ²={sta:.3f}  p={pa:.4g}  → {verdict}")
        return dict(a=a_lbl, b=b_lbl, a_right=ba, b_right=ca, chi2=sta, p=pa)

    print()
    mc_fc = _mc("fused-vs-CNN", "CNN", cnn_pred, "fused", fused_pred)
    mc_fg = _mc("fused-vs-geom", "geom", geom_pred, "fused", fused_pred)
    mc_cg = _mc("CNN-vs-geom", "CNN", cnn_pred, "geom", geom_pred)
    b, c, st, pval = mc_fc["a_right"], mc_fc["b_right"], mc_fc["chi2"], mc_fc["p"]

    ckpt = dict(fuse=fused_cfg, geom_source=fcfg["geom_source"],
                model_cfg=mcfg, k_neighbors=kN, cnn_oof=cnn_oof,
                cnn_train=m_cnn, fused_train=m_fused,
                mcnemar=dict(b=b, c=c, chi2=st, p=pval),
                mcnemar_pairs=dict(fused_vs_cnn=mc_fc, fused_vs_geom=mc_fg,
                                   cnn_vs_geom=mc_cg),
                calibrate=cal_method, calibrate_components=sorted(cal_set),
                cal_cnn=cal_cnn, cal_geom=cal_geom,
                row_consistency=fcfg.get("row_consistency"))
    ckpt_path = os.path.join(run_dir, "fuse.pkl")
    with open(ckpt_path, "wb") as fh:
        pickle.dump(ckpt, fh)
    print(f"saved fusion ckpt → {ckpt_path}")

    # Per-cell OOF predictions — enables offline fusion analysis (adaptive
    # weighting, oracle ceilings) without re-retraining the per-fold CNNs.
    oof_path = os.path.join(run_dir, "oof_preds.npz")
    np.savez_compressed(
        oof_path, y=y, groups=tr["groups"],
        pc_raw=pc_raw, pg_raw=pg_raw, pc_cal=pc, pg_cal=pg, p_fused=p_fused,
        names=np.asarray([r[0] for r in tr["ref"]], dtype=object),
        cell_ids=np.asarray([r[1] for r in tr["ref"]], dtype=np.int64),
        fused_w=float(fused_cfg.get("weight", np.nan)),
        fused_th=float(fused_cfg.get("threshold", 0.5)))
    print(f"saved OOF per-cell preds → {oof_path}")

    # held-out test: refit geom on ALL train, freeze fusion, score test segs.
    if cfg["data"].get("seg_test_dir"):
        print(f"\nassembling test segs ({cfg['data']['seg_test_dir']}) …")
        te = assemble_segs(cfg["data"]["seg_test_dir"], k_neighbors=kN,
                           need_cnn=True, need_gt=True, group_mode=gm)
        if fcfg["geom_source"] == "rule":
            pg_t = rule_proba_ihc(te["feats"], te["groups"])
        else:
            gm = fit_geom(tr["feats"], tr["y"], mcfg)
            pg_t = proba_ihc(gm, te["feats"])
        pc_t_raw, pg_t_raw, y_t = te["p_cnn"], pg_t, te["y"]
        # apply the *same* calibrators fit on OOF train.
        pc_t = apply_calibrator(cal_cnn, pc_t_raw)
        pg_t = apply_calibrator(cal_geom, pg_t_raw)
        if fused_cfg["method"] == "stack":
            coef = np.array(fused_cfg["coef"])
            z = np.column_stack([pc_t, pg_t]) @ coef + fused_cfg["intercept"]
            pf_t = 1.0 / (1.0 + np.exp(-z))
        else:
            pf_t = _fuse_mean(pc_t, pg_t, fused_cfg["weight"])
        print(f"\n=== HELD-OUT TEST ({len(y_t)} cells) ===")
        print(f"  ECE (test)  CNN  {ece(pc_t_raw, y_t):.4f} → "
              f"{ece(pc_t, y_t):.4f}")
        print(f"  ECE (test)  geom {ece(pg_t_raw, y_t):.4f} → "
              f"{ece(pg_t, y_t):.4f}")
        print(f"  ECE (test)  fused {ece(pf_t, y_t):.4f}")
        test_th = fused_cfg.get("threshold", 0.5)
        for tag, p, th in (("CNN alone", pc_t, 0.5),
                            ("geom alone", pg_t, 0.5),
                            (f"FUSED @th={test_th:.3f}", pf_t, test_th)):
            cm = cm_from(y_t, np.where(p >= th, IHC, OHC))
            print(f"\n[{tag}]\n{fmt_cm(cm)}")
        bt, ct, stt, pt = mcnemar(y_t, np.where(pc_t >= 0.5, IHC, OHC),
                                  np.where(pf_t >= test_th, IHC, OHC))
        print(f"\nMcNemar (test) fused-vs-CNN: CNN-only-right={bt}  "
              f"fused-only-right={ct}  χ²={stt:.3f}  p={pt:.4g}")
    return run_dir


# ── predict (write back into one seg) ───────────────────────────────────────────

_PERP_IDX = FEATURE_NAMES.index("perp_signed")


def row_consistency_refine(cids, feats, fused_map, fused_prob, *,
                           conf_anchor=0.85, conf_override=0.60, k_anchor=15):
    """Per-image row-consistency post-pass over the fused labels.

    IHC and OHC cells separate almost perfectly by signed perpendicular
    offset from the organ-of-Corti axis (`perp_signed`). The residual fused
    errors are cells sitting unambiguously in one row that got the other
    label at low confidence. This re-anchors the two perp bands on the
    *confident* fused calls (≥`conf_anchor`) and, for any cell whose own
    fused label is uncertain (<`conf_override`) yet whose perp position is
    nearer the other band, flips it to match.

    Confidence-gated, so it never touches a confident fused call — on the
    CLC curator-correction set this recovered 40/68 errors with **zero**
    cells flipped the wrong way. No-op (empty `overrides`) when either band
    lacks ≥3 confident anchors. Returns (refined_map, refined_prob,
    overrides) with overrides = {cid: previous_label}.
    """
    perp = {int(c): float(f[_PERP_IDX]) for c, f in zip(cids, feats)}
    aI = np.array([perp[c] for c in fused_map if c in perp
                   and fused_map[c] == "IHC"
                   and fused_prob.get(c, 0.0) >= conf_anchor])
    aO = np.array([perp[c] for c in fused_map if c in perp
                   and fused_map[c] == "OHC"
                   and fused_prob.get(c, 0.0) >= conf_anchor])
    if len(aI) < 3 or len(aO) < 3:
        return dict(fused_map), dict(fused_prob), {}
    refined, rprob, overrides = dict(fused_map), dict(fused_prob), {}
    for c in fused_map:
        if c not in perp:
            continue
        p = perp[c]
        dI = np.sort(np.abs(aI - p))[:min(k_anchor, len(aI))].mean()
        dO = np.sort(np.abs(aO - p))[:min(k_anchor, len(aO))].mean()
        geo = "IHC" if dI < dO else "OHC"
        if geo != fused_map[c] and fused_prob.get(c, 1.0) < conf_override:
            overrides[c] = fused_map[c]
            refined[c] = geo
            margin = abs(dI - dO) / (dI + dO + 1e-9)
            rprob[c] = float(0.5 + 0.5 * min(margin, 1.0))
    return refined, rprob, overrides


def predict_seg_geom(geom_ckpt, seg_path, *, fuse_ckpt=None, write=False):
    """Score every instance in one seg.npy with the geom model (and,
    given a fusion ckpt + the CNN keys, the fused decision). Writes
    class_map_geom / class_prob_geom / geom_flag (+ _fused) — masks and
    the CNN's keys are left untouched. For a whole directory use
    `predict_seg_geom_dir` (loads the ckpts once)."""
    with open(geom_ckpt, "rb") as fh:
        gk = pickle.load(fh)
    fk = None
    if fuse_ckpt:
        with open(fuse_ckpt, "rb") as fh:
            fk = pickle.load(fh)
    return _score_seg_geom(gk, fk, seg_path, write=write)


def predict_seg_geom_dir(geom_ckpt, data_dir, *, fuse_ckpt=None, write=True,
                         include_augmented=False):
    """Batched geom (+ fusion) inference over a directory.

    Loads the geom and fuse ckpts **once** and scores every seg in-process
    — the geom analogue of `ihc_ohc_pipeline.cnn_predict_dir`, replacing the
    old per-seg subprocess loop (which reloaded both pickles and re-imported
    sklearn for each of N segs). Sidecars only; the dataset segs are
    untouched."""
    with open(geom_ckpt, "rb") as fh:
        gk = pickle.load(fh)
    fk = None
    if fuse_ckpt:
        with open(fuse_ckpt, "rb") as fh:
            fk = pickle.load(fh)
    n = 0
    for sp, _ in iter_seg_files(data_dir, include_augmented):
        _score_seg_geom(gk, fk, sp, write=write)
        n += 1
    print(f"  geom write-back: {n} sidecars (seg files untouched) in {data_dir}")
    return n


def _score_seg_geom(gk, fk, seg_path, *, write=False):
    """Score one seg with already-loaded geom (`gk`) and fuse (`fk` or None)
    ckpts — the per-seg core shared by `predict_seg_geom` and
    `predict_seg_geom_dir`."""
    model = (gk["scaler"], gk["clf"])
    kN = gk["model_cfg"]["k_neighbors"]

    seg = np.load(seg_path, allow_pickle=True).item()
    # Option A: exclude reject-pass off-band masks from the centreline / kNN
    # fit and from scoring, so they neither skew the geometry nor get a label.
    excl = active_reject_ids(seg_path, seg)
    cids, feats, flags, _ = geom_features_for_seg(
        seg, k_neighbors=kN, exclude_ids=excl)
    if excl:
        print(f"  hair-cell reject: excluding {len(excl)} off-band mask(s)")
    if not cids:
        print("no instances")
        return {}
    pg = proba_ihc(model, feats)
    geom_map = {int(c): (CLASS_NAMES[IHC] if p >= 0.5 else CLASS_NAMES[OHC])
                for c, p in zip(cids, pg)}
    geom_prob = {int(c): float(max(p, 1 - p)) for c, p in zip(cids, pg)}
    flag_map = {int(c): int(f) for c, f in zip(cids, flags)}

    fused_map = fused_prob = None
    row_override = {}
    if fk is not None:
        pred = load_pred(seg_path, seg)
        pc = np.array([_cnn_pihc(pred, int(c)) if pred else None
                       for c in cids], float)
        if np.isnan(pc).any():
            print("  fusion skipped: CNN class_prob missing for some cells "
                  "(run ihc_ohc_classifier.py predict --write first)")
        else:
            # apply the calibrators frozen at fuse-time (if the ckpt has
            # them; older ckpts without are honoured by passing through).
            cc, cg = fk.get("cal_cnn"), fk.get("cal_geom")
            pc_use = apply_calibrator(cc, pc) if cc is not None else pc
            pg_use = apply_calibrator(cg, pg) if cg is not None else pg
            f = fk["fuse"]
            if f["method"] == "stack":
                z = (np.column_stack([pc_use, pg_use]) @ np.array(f["coef"])
                     + f["intercept"])
                pf = 1.0 / (1.0 + np.exp(-z))
            else:
                pf = f["weight"] * pc_use + (1 - f["weight"]) * pg_use
            th = float(f.get("threshold", 0.5))  # tuned at fuse-time
            fused_map = {int(c): (CLASS_NAMES[IHC] if p >= th
                                  else CLASS_NAMES[OHC])
                         for c, p in zip(cids, pf)}
            fused_prob = {int(c): float(max(p, 1 - p))
                          for c, p in zip(cids, pf)}
            # row-consistency post-pass (per-image; params ride in the
            # fuse ckpt so deployed/GUI inference applies the same rule)
            rc = fk.get("row_consistency") or {}
            if rc.get("enabled") and fused_map:
                fused_map, fused_prob, row_override = row_consistency_refine(
                    cids, feats, fused_map, fused_prob,
                    conf_anchor=rc.get("conf_anchor", 0.85),
                    conf_override=rc.get("conf_override", 0.60),
                    k_anchor=rc.get("k_anchor", 15))
                if row_override:
                    print(f"  row-consistency: flipped {len(row_override)} "
                          f"low-confidence fused label(s) to match perp band")

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
        extra = {"row_override": row_override} if row_override else {}
        pp = update_pred(seg_path, class_map_geom=geom_map,
                         class_prob_geom=geom_prob, geom_flag=flag_map,
                         class_map_fused=fused_map,
                         class_prob_fused=fused_prob, **extra)
        print(f"  wrote class_map_geom / class_prob_geom / geom_flag"
              + (" / class_map_fused / class_prob_fused" if fused_map else "")
              + (f" / row_override({len(row_override)})" if row_override else "")
              + f" → {os.path.basename(pp)} (sidecar; seg untouched)")
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
        if name in ("train", "sweep", "fuse"):
            s.add_argument("--run_dir", default=None,
                           help="override the auto-stamped run dir "
                           "(used by ihc_ohc_pipeline.py to group steps)")
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
    if args.cmd == "train":
        train(cfg, run_dir=args.run_dir)
    elif args.cmd == "sweep":
        sweep(cfg, run_dir=args.run_dir)
    elif args.cmd == "fuse":
        fuse(cfg, run_dir=args.run_dir)
    elif args.cmd == "cv":
        cross_validate(cfg)
    else:  # rule
        rule_cmd(cfg)


if __name__ == "__main__":
    main()
