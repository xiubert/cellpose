"""
Phase 2 — tiny CNN for IHC vs OHC classification of Cellpose instances.

Input  : (2, S, S)  — MYO7A crop + target-instance mask, from ihc_ohc_crops.py
Output : 2-class logits (0=IHC, 1=OHC)

All hyperparameters live in a YAML config (see ihc_ohc_config.yaml), not in
CLI flags, so runs are reproducible and sweeps are declarative.

Subcommands (each takes --config <yaml>)
----------------------------------------
  train    fit TinyHCNet on the train crops .npz with one image-level
           train/val split; write checkpoint + confusion matrix + montage.

  cv       GroupKFold cross-validation over source images — leak-free
           generalisation estimate (mean ± std bal_acc across folds).

  sweep    cartesian product over the `sweep:` lists in the config, each
           combination scored by CV; prints a ranked leaderboard so you can
           pick the config to `train`.

  predict  run a trained checkpoint over a Cellpose _seg.npy: crop every
           instance with the *same* expansion code, classify, and write
           `class_map_pred` / `class_prob` back into the seg dict.

Dependencies (beyond the cellpose env): PyYAML (config) and scikit-learn
(sklearn.model_selection.GroupKFold) — both must be installed in the
container; see ihc_ohc.md.

Usage (inside the cellpose container; /helpers is the host helpers dir)
-----------------------------------------------------------------------
  python /helpers/ihc_ohc_crops.py --data_dir .../traintest/train --out /helpers/crops_train.npz
  python /helpers/ihc_ohc_crops.py --data_dir .../traintest/test  --out /helpers/crops_test.npz

  python /helpers/ihc_ohc_classifier.py sweep --config /helpers/ihc_ohc_config.yaml
  python /helpers/ihc_ohc_classifier.py cv    --config /helpers/ihc_ohc_config.yaml
  python /helpers/ihc_ohc_classifier.py train --config /helpers/ihc_ohc_config.yaml
  python /helpers/ihc_ohc_classifier.py predict \
      --ckpt /helpers/ihc_ohc_run/best.pt \
      --seg  /data/.../000_cunningham_mouse_confocal_myo7a_seg.npy --write
"""

import argparse
import copy
import itertools
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF

from ihc_ohc_crops import (
    CLASS_NAMES, extract_cell_crop, load_image_plane, resolve_class_map,
    seg_stem, tif_for_seg, update_pred,
)


# ── config ─────────────────────────────────────────────────────────────────────

# Single source of truth for every hyperparameter. A user YAML is deep-merged
# over this, so partial configs are valid and unknown keys are rejected. The
# `train` values are the tuned defaults derived from the smoke-test analysis.
DEFAULT_CONFIG = {
    "data": {
        "train_npz": "/helpers/crops_train.npz",
        "test_npz": "/helpers/crops_test.npz",
        "out_dir": "/helpers/ihc_ohc_run",
    },
    "train": {
        "epochs": 40,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 4,
        "select_ema": 0.5,        # EMA factor for the selection metric; 0 = raw
        "sampler": "balanced",    # balanced | random
        "loss": "ce",             # ce | focal
        "focal_gamma": 1.5,
        "class_weight": "none",   # none | balanced
        "patience": 15,
        "workers": 0,             # 0 = load in main process; the cellpose
                                  # container's /dev/shm (63 MB) is too small
                                  # for DataLoader worker IPC (bus error)
        "seed": 0,
    },
    "cv": {
        "folds": 5,               # >=2 → GroupKFold; <=1 → single split
        "val_frac": 0.2,          # used only when folds <= 1
    },
    # Any train.* key may be a list here → cartesian product, CV-scored.
    "sweep": {},
}


def _deep_merge(base, override, path=""):
    """Recursively merge override into a copy of base; reject unknown keys.

    `sweep` is free-form (arbitrary train.* keys) so its contents are not
    validated against the schema.
    """
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
    """Load a YAML config, deep-merged over DEFAULT_CONFIG and validated."""
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    cfg = _deep_merge(DEFAULT_CONFIG, user)
    # sweep keys must name real train.* hyperparameters
    for k in cfg.get("sweep", {}):
        if k not in DEFAULT_CONFIG["train"]:
            raise KeyError(f"sweep key '{k}' is not a train.* hyperparameter")
    return cfg


def print_config(cfg, tag="config"):
    """Echo the fully-resolved config so the actual run is never ambiguous."""
    print(f"=== {tag} ===")
    print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False).rstrip())
    print("=" * (len(tag) + 8))


# ── model ──────────────────────────────────────────────────────────────────────

def _gn(c, max_groups=8):
    """GroupNorm with the largest power-of-two group count that divides c.

    GroupNorm (not BatchNorm) because batches here are small and class-
    imbalanced — BN's running stats diverge between train and eval and
    produced the violent val-loss oscillation in the smoke test.
    """
    g = max_groups
    while c % g != 0:
        g //= 2
    return nn.GroupNorm(max(g, 1), c)


def _block(c_in, c_out):
    """Two 3×3 conv → GroupNorm → ReLU, then 2× max-pool."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        _gn(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        _gn(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TinyHCNet(nn.Module):
    """~50k-param CNN: 3 conv blocks → global average pool → linear.

    Deliberately small — IHC/OHC on a context crop is an easy task and
    capacity is not the limiting factor on the Cunningham set.
    """

    def __init__(self, in_ch=2, n_classes=2, widths=(16, 32, 64), p_drop=0.3):
        super().__init__()
        c = in_ch
        blocks = []
        for w in widths:
            blocks.append(_block(c, w))
            c = w
        self.features = nn.Sequential(*blocks)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # global average pool
        return self.fc(self.drop(x))


# ── dataset + augmentation ─────────────────────────────────────────────────────

def compute_norm_stats(crops):
    """Per-image-channel mean/std (channel 0 only; mask channel stays 0–1)."""
    img = crops[:, 0]
    return float(img.mean()), float(img.std() + 1e-6)


class CropDataset(Dataset):
    """Crops + labels with optional on-the-fly augmentation.

    Order: photometric jitter (image channel, raw) → standardise image
    channel → geometric (rotation 0–360°, flips, ±translate, applied to both
    channels) → optional random erasing.
    """

    def __init__(self, crops, labels, mean, std, *, train=False,
                 max_translate=3, p_erase=0.25, seed=0):
        self.crops = crops
        self.labels = labels.astype(np.int64)
        self.mean, self.std = mean, std
        self.train = train
        self.max_translate = max_translate
        self.p_erase = p_erase
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.crops[i, 0].astype(np.float32).copy()
        msk = self.crops[i, 1].astype(np.float32).copy()

        if self.train:  # photometric — image channel only, in intensity space
            img *= self.rng.uniform(0.8, 1.2)
            hi = float(img.max()) or 1.0
            img = (np.clip(img, 0, None) / hi) ** self.rng.uniform(0.8, 1.25) * hi

        img = (img - self.mean) / self.std
        x = torch.from_numpy(np.stack([img, msk], 0))

        if self.train:
            if self.rng.random() < 0.5:
                x = TF.hflip(x)
            if self.rng.random() < 0.5:
                x = TF.vflip(x)
            angle = float(self.rng.uniform(0, 360))
            t = self.max_translate
            tx = int(self.rng.integers(-t, t + 1)) if t else 0
            ty = int(self.rng.integers(-t, t + 1)) if t else 0
            x = TF.affine(x, angle=angle, translate=(tx, ty), scale=1.0,
                          shear=[0.0], interpolation=TF.InterpolationMode.BILINEAR,
                          fill=0.0)  # 0 ≈ background (image is mean-centred)
            if self.rng.random() < self.p_erase:
                s = int(self.rng.integers(4, 12))
                H, W = x.shape[-2:]
                yy = int(self.rng.integers(0, H - s))
                xx = int(self.rng.integers(0, W - s))
                x[:, yy:yy + s, xx:xx + s] = 0.0

        return x.float(), int(self.labels[i])


# ── data prep ──────────────────────────────────────────────────────────────────

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    meta = d["meta"][0] if "meta" in d else {}
    return (d["crops"].astype(np.float32), d["labels"].astype(np.int64),
            d["groups"].astype(np.int64), list(d["image_names"]), meta)


def group_split(groups, val_frac, seed):
    """Split *by source image* so no cell leaks across train/val."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_g = set(uniq[:n_val].tolist())
    is_val = np.array([g in val_g for g in groups])
    return ~is_val, is_val


def class_weights(labels, n_classes=2):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def make_sampler(labels, n_classes=2):
    """WeightedRandomSampler giving ~uniform class frequency per batch.

    Balanced batches are the main imbalance lever here: they lift minority
    recall *and* feed GroupNorm a class-balanced view every step. Use this
    instead of (not on top of) loss weighting to avoid double-correcting.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1)
    sample_w = inv[labels]
    return WeightedRandomSampler(
        torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(labels), replacement=True)


def make_scheduler(opt, epochs, warmup):
    """Linear warmup → cosine decay, stepped once per epoch.

    Replaces ReduceLROnPlateau, which keyed on a val-loss signal too noisy
    (93 images → ~19 val) to ever fire usefully.
    """
    def lr_lambda(ep):  # ep is the 0-indexed epoch count from LambdaLR
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        prog = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class FocalLoss(nn.Module):
    """Multiclass focal loss — down-weights easy examples (γ), optional α.

    Second minority lever: with balanced sampling already on, leave α=None
    (weight) and lean on γ to focus learning on the hard IHC/OHC boundary.
    """

    def __init__(self, gamma=1.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def build_criterion(tcfg, train_labels, device):
    """CE or focal, with class weighting only if explicitly requested."""
    weight = None
    if tcfg["class_weight"] == "balanced":
        weight = class_weights(train_labels).to(device)
    if tcfg["loss"] == "focal":
        return FocalLoss(gamma=tcfg["focal_gamma"], weight=weight)
    return nn.CrossEntropyLoss(weight=weight)


# ── train / eval ───────────────────────────────────────────────────────────────

def metrics_from_cm(cm):
    """acc, balanced accuracy, macro-F1 from a confusion matrix."""
    acc = cm.trace() / max(cm.sum(), 1)
    recalls, f1s = [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        rec = tp / max(cm[i].sum(), 1)
        prec = tp / max(cm[:, i].sum(), 1)
        recalls.append(rec)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return dict(acc=acc, bal_acc=float(np.mean(recalls)),
                macro_f1=float(np.mean(f1s)))


@torch.no_grad()
def evaluate(model, loader, device, n_classes=2):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    loss_sum = n = 0
    crit = nn.CrossEntropyLoss()
    preds_all, ys_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * len(y)
        n += len(y)
        p = logits.argmax(1)
        for t, q in zip(y.cpu().numpy(), p.cpu().numpy()):
            cm[t, q] += 1
        preds_all.append(p.cpu().numpy())
        ys_all.append(y.cpu().numpy())
    return (loss_sum / max(n, 1), cm,
            np.concatenate(ys_all), np.concatenate(preds_all))


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


def fit(crops, labels, tr_idx, va_idx, tcfg, device, *, verbose=True):
    """Core training loop for one train/val split.

    Norm stats are computed from this split's *train* part only (no leakage),
    so every CV fold gets its own normalisation. Returns the best-epoch
    weights plus history and the selected-epoch val metrics.
    """
    mean, std = compute_norm_stats(crops[tr_idx])
    y_tr = labels[tr_idx]

    ds_tr = CropDataset(crops[tr_idx], y_tr, mean, std, train=True,
                        seed=tcfg["seed"])
    if tcfg["sampler"] == "balanced":
        dl_tr = DataLoader(ds_tr, batch_size=tcfg["batch_size"],
                           sampler=make_sampler(y_tr),
                           num_workers=tcfg["workers"], drop_last=True)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=tcfg["batch_size"], shuffle=True,
                           num_workers=tcfg["workers"], drop_last=True)
    dl_va = DataLoader(
        CropDataset(crops[va_idx], labels[va_idx], mean, std, train=False),
        batch_size=tcfg["batch_size"], shuffle=False, num_workers=tcfg["workers"])

    model = TinyHCNet(in_ch=crops.shape[1]).to(device)
    crit = build_criterion(tcfg, y_tr, device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    sched = make_scheduler(opt, tcfg["epochs"], tcfg["warmup_epochs"])

    # Selection on EMA-smoothed balanced accuracy (maximise). Raw val
    # bal_acc spikes epoch-to-epoch on the ~19-image val set, so picking the
    # single best epoch is fragile; the EMA tracks the true level instead.
    best_score, best_state, best_ep, bad = -1.0, None, 0, 0
    ema = None
    a = tcfg["select_ema"]
    history = []
    for ep in range(1, tcfg["epochs"] + 1):
        model.train()
        tl = tn = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            tn += len(y)
        sched.step()
        vl, cm, _, _ = evaluate(model, dl_va, device)
        m = metrics_from_cm(cm)
        score = m["bal_acc"]
        ema = score if ema is None else (1 - a) * ema + a * score
        sel = score if a <= 0 else ema  # a<=0 → raw (smoothing off)
        history.append(dict(epoch=ep, lr=opt.param_groups[0]["lr"],
                            train_loss=tl / max(tn, 1), val_loss=vl,
                            val_acc=m["acc"], val_bal_acc=score,
                            val_bal_acc_ema=ema, val_macro_f1=m["macro_f1"]))
        if verbose:
            print(f"ep {ep:3d}  train {tl/max(tn,1):.4f}  val {vl:.4f}  "
                  f"acc {m['acc']:.4f}  bal_acc {score:.4f}  ema {ema:.4f}  "
                  f"f1 {m['macro_f1']:.4f}"
                  + ("  *" if sel > best_score + 1e-4 else ""))
        if sel > best_score + 1e-4:
            best_score, best_ep, bad = sel, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                if verbose:
                    print(f"early stop (no smoothed bal_acc gain for "
                          f"{tcfg['patience']} epochs)")
                break

    return dict(best_state=best_state, mean=mean, std=std, history=history,
                best_ep=best_ep, best_score=best_score,
                best_metrics=history[best_ep - 1])


def train(cfg):
    """One image-level split → fit → checkpoint + held-out test report."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = cfg["train"]
    out_dir = cfg["data"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    crops, labels, groups, names, meta = load_npz(cfg["data"]["train_npz"])
    tr, va = group_split(groups, cfg["cv"]["val_frac"], tcfg["seed"])
    print(f"train crops {tr.sum()} | val crops {va.sum()} | "
          f"images {len(np.unique(groups))}")
    print(f"TinyHCNet | sampler={tcfg['sampler']} loss={tcfg['loss']} "
          f"class_weight={tcfg['class_weight']}")

    r = fit(crops, labels, tr, va, tcfg, device, verbose=True)
    bm = r["best_metrics"]
    print(f"best epoch {r['best_ep']}  smoothed bal_acc {r['best_score']:.4f}  "
          f"(raw bal_acc {bm['val_bal_acc']:.4f}  macro_f1 {bm['val_macro_f1']:.4f})")

    ckpt = dict(state_dict=r["best_state"], mean=r["mean"], std=r["std"],
                class_names=CLASS_NAMES, in_ch=int(crops.shape[1]),
                crop_meta=meta if isinstance(meta, dict) else dict(meta),
                train_cfg=dict(tcfg, best_epoch=r["best_ep"],
                               best_sel_score=r["best_score"]))
    ckpt_path = os.path.join(out_dir, "best.pt")
    torch.save(ckpt, ckpt_path)
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump(r["history"], fh, indent=2)
    print(f"\nsaved checkpoint → {ckpt_path}")

    if cfg["data"]["test_npz"]:
        model = TinyHCNet(in_ch=crops.shape[1]).to(device)
        model.load_state_dict(r["best_state"])
        tc, tlab, _, _, _ = load_npz(cfg["data"]["test_npz"])
        dl_te = DataLoader(CropDataset(tc, tlab, r["mean"], r["std"], train=False),
                           batch_size=tcfg["batch_size"])
        loss, cm, ys, preds = evaluate(model, dl_te, device)
        print(f"\nTEST  loss {loss:.4f}\n{fmt_cm(cm)}")
        _save_misclassified(os.path.join(out_dir, "test_misclassified.png"),
                            tc, ys, preds)


# ── cross-validation & sweep ────────────────────────────────────────────────────

def cross_validate(cfg, *, verbose=True):
    """GroupKFold over source images → leak-free generalisation estimate.

    Returns the per-fold selected-epoch metrics and their mean/std. Folds
    split on `groups` (source image), so no cell leaks between train and val.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = cfg["train"]
    folds = int(cfg["cv"]["folds"])
    crops, labels, groups, names, meta = load_npz(cfg["data"]["train_npz"])

    if folds <= 1:  # degenerate "CV" = the single split train() uses
        splits = [tuple(np.where(m)[0] for m in
                        group_split(groups, cfg["cv"]["val_frac"], tcfg["seed"]))]
    else:
        gkf = GroupKFold(n_splits=folds)
        splits = list(gkf.split(crops, labels, groups))

    rows = []
    for k, (tr_idx, va_idx) in enumerate(splits, 1):
        ng = len(np.unique(groups[va_idx]))
        r = fit(crops, labels, tr_idx, va_idx, tcfg, device, verbose=False)
        bm = r["best_metrics"]
        rows.append(dict(fold=k, val_images=ng, best_ep=r["best_ep"],
                         acc=bm["val_acc"], bal_acc=bm["val_bal_acc"],
                         macro_f1=bm["val_macro_f1"]))
        if verbose:
            print(f"  fold {k}/{len(splits)}  val_imgs {ng:2d}  "
                  f"ep {r['best_ep']:3d}  acc {bm['val_acc']:.4f}  "
                  f"bal_acc {bm['val_bal_acc']:.4f}  "
                  f"f1 {bm['val_macro_f1']:.4f}")

    def ms(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std())}

    agg = {k: ms(k) for k in ("acc", "bal_acc", "macro_f1")}
    if verbose:
        print(f"CV {len(splits)}-fold  "
              f"bal_acc {agg['bal_acc']['mean']:.4f} ± {agg['bal_acc']['std']:.4f}  "
              f"acc {agg['acc']['mean']:.4f} ± {agg['acc']['std']:.4f}  "
              f"macro_f1 {agg['macro_f1']['mean']:.4f} ± {agg['macro_f1']['std']:.4f}")
    return dict(folds=rows, mean_std=agg)


def _sweep_grid(sweep):
    """sweep dict of {key: [candidates]} → list of override dicts (product)."""
    if not sweep:
        return [{}]
    keys = list(sweep)
    axes = [v if isinstance(v, list) else [v] for v in sweep.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes)]


def sweep(cfg):
    """CV-score every point in the sweep grid; print a ranked leaderboard."""
    grid = _sweep_grid(cfg.get("sweep", {}))
    out_dir = cfg["data"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"sweep: {len(grid)} configurations × {cfg['cv']['folds']}-fold CV")

    results = []
    for i, override in enumerate(grid, 1):
        trial = copy.deepcopy(cfg)
        trial["train"].update(override)
        tag = ", ".join(f"{k}={v}" for k, v in override.items()) or "baseline"
        print(f"\n[{i}/{len(grid)}] {tag}")
        cv = cross_validate(trial, verbose=True)
        agg = cv["mean_std"]
        results.append(dict(override=override,
                            bal_acc_mean=agg["bal_acc"]["mean"],
                            bal_acc_std=agg["bal_acc"]["std"],
                            acc_mean=agg["acc"]["mean"],
                            macro_f1_mean=agg["macro_f1"]["mean"],
                            folds=cv["folds"]))

    results.sort(key=lambda r: r["bal_acc_mean"], reverse=True)
    print("\n=== sweep leaderboard (by mean CV bal_acc) ===")
    for rank, r in enumerate(results, 1):
        tag = ", ".join(f"{k}={v}" for k, v in r["override"].items()) or "baseline"
        print(f"  {rank:2d}. bal_acc {r['bal_acc_mean']:.4f} ± "
              f"{r['bal_acc_std']:.4f}  f1 {r['macro_f1_mean']:.4f}  | {tag}")
    best = results[0]
    print(f"\nbest: {best['override'] or 'baseline'}  "
          f"→ set these in train: and run `train --config`")
    with open(os.path.join(out_dir, "sweep_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"saved → {os.path.join(out_dir, 'sweep_results.json')}")


def _save_misclassified(path, crops, ys, preds, limit=40):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("misclassified montage skipped (matplotlib not installed)")
        return
    wrong = np.where(ys != preds)[0]
    if wrong.size == 0:
        print("no misclassified test crops")
        return
    wrong = wrong[:limit]
    cols = 8
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for j, k in enumerate(wrong):
        r, c = divmod(j, cols)
        img = crops[k, 0]
        lo, hi = np.percentile(img, (1, 99))
        axes[r][c].imshow(np.clip((img - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray")
        axes[r][c].set_title(f"t={CLASS_NAMES[ys[k]]} p={CLASS_NAMES[preds[k]]}",
                             fontsize=6)
    fig.suptitle(f"misclassified ({wrong.size} shown)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"misclassified montage → {path}")


# ── inference ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TinyHCNet(in_ch=ck.get("in_ch", 2)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def predict_crops(model, crops, mean, std, device="cpu", batch=256):
    """crops (N,2,S,S) raw → (pred_idx (N,), prob (N, n_classes))."""
    preds, probs = [], []
    for i in range(0, len(crops), batch):
        x = crops[i:i + batch].astype(np.float32).copy()
        x[:, 0] = (x[:, 0] - mean) / std
        logits = model(torch.from_numpy(x).to(device))
        p = torch.softmax(logits, 1)
        probs.append(p.cpu().numpy())
        preds.append(p.argmax(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def predict_seg(ckpt_path, seg_path, *, write=False, device=None):
    """Classify every instance in a Cellpose _seg.npy.

    Reuses the Phase-1 crop expansion (same geometry as training). Returns
    {cell_id: (class_name, prob)}; with write=True also stores
    `class_map_pred` and `class_prob` back into the seg dict.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(ckpt_path, device)
    cm_meta = ck.get("crop_meta", {}) or {}
    out_size = int(cm_meta.get("out_size", 64))
    pad_frac = float(cm_meta.get("pad_frac", 0.5))
    pad_px = cm_meta.get("pad_px")
    soft_mask = bool(cm_meta.get("soft_mask", True))
    channel = int(cm_meta.get("channel", 1))

    seg = np.load(seg_path, allow_pickle=True).item()
    masks = seg["masks"]
    plane = load_image_plane(seg, tif_for_seg(seg_path), channel)
    fill = float(plane.mean())

    ids = [int(i) for i in np.unique(masks) if i != 0]
    crops, valid = [], []
    for cid in ids:
        crop = extract_cell_crop(plane, masks, cid, out_size=out_size,
                                 pad_frac=pad_frac, pad_px=pad_px,
                                 pad_value=fill, soft_mask=soft_mask)
        if crop is not None:
            crops.append(crop)
            valid.append(cid)
    if not crops:
        print("no instances to classify")
        return {}

    preds, probs = predict_crops(model, np.stack(crops), ck["mean"], ck["std"], device)
    result = {cid: (CLASS_NAMES[p], float(pr[p]))
              for cid, p, pr in zip(valid, preds, probs)}

    n_ihc = sum(v[0] == "IHC" for v in result.values())
    print(f"{os.path.basename(seg_path)}: {len(result)} cells  "
          f"IHC {n_ihc} / OHC {len(result) - n_ihc}")

    # accuracy vs. ground truth, when available
    gt, src = resolve_class_map(seg, seg_path, os.path.dirname(seg_path))
    if gt:
        ok = tot = 0
        for cid, (cls, _) in result.items():
            if cid in gt:
                tot += 1
                ok += (gt[cid] == cls)
        if tot:
            print(f"  vs ground truth [{src}]: acc {ok/tot:.4f}  ({ok}/{tot})")

    if write:
        pp = update_pred(
            seg_path,
            class_map_pred={cid: v[0] for cid, v in result.items()},
            class_prob={cid: v[1] for cid, v in result.items()})
        print(f"  wrote class_map_pred / class_prob → "
              f"{os.path.basename(pp)} (sidecar; seg untouched)")
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="IHC/OHC tiny-CNN classifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, helptext in (
        ("train", "one split → checkpoint + held-out test report"),
        ("cv",    "GroupKFold cross-validation (generalisation estimate)"),
        ("sweep", "CV-score the sweep grid → ranked leaderboard"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--config", required=True,
                       help="YAML config (see ihc_ohc_config.yaml)")

    q = sub.add_parser("predict", help="classify instances in a _seg.npy")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--seg", required=True)
    q.add_argument("--write", action="store_true",
                   help="store class_map_pred / class_prob back into the seg.npy")
    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "predict":
        predict_seg(args.ckpt, args.seg, write=args.write)
        return
    cfg = load_config(args.config)
    print_config(cfg, tag=f"{args.cmd} config ({args.config})")
    {"train": train, "cv": cross_validate, "sweep": sweep}[args.cmd](cfg)


if __name__ == "__main__":
    main()
