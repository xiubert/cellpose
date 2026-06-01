"""
Online (on-the-fly) photometric + occlusion augmentation for Cellpose-SAM
fine-tuning — a drop-in replacement for the photometric half of augment.py.

WHY THIS EXISTS
---------------
cellpose's training loop already applies the *geometric* augmentations that
augment.py wrote to disk: ``random_rotate_and_resize`` (cellpose/transforms.py)
does a continuous random rotation (0-2pi), random flip, random scale
(``scale_range``) and a random ``bsize`` crop, fresh every epoch. So the D4
``_SV_rot90/_fliph/...`` copies add no geometric information — only sampling
multiplicity and disk bloat, and they can be retired for training.

What cellpose does NOT do is any *photometric* or *occlusion* augmentation.
This module adds it ONLINE, with zero new files on disk. ``make(cfg)`` returns
a callable that ``train_seg(..., img_transform=...)`` applies to each augmented
TRAIN image batch ``imgi`` (B, C, bsize, bsize) float32, right after the
geometry/normalization and before the forward pass. It returns a new batch of
the same shape and never touches the labels, so mask labels stay pixel-exact.

The hook itself is a tiny generic param in cellpose/train.py; all augmentation
logic lives here (see the train.py ``img_transform`` docstring).

CONFIG (trainer.yaml ``augment:`` block; every section optional)
----------------------------------------------------------------
augment:
  seed: 0
  intensity:   {p: 0.5, scale: [0.75, 1.25], gamma: [0.8, 1.25]}
  cutout:      {p: 0.5, n: [1, 3], size_frac: [0.05, 0.15], fill: mean}
  gauss_noise: {p: 0.3, sigma_frac: 0.05}
  gauss_blur:  {p: 0.2, sigma: [0.5, 1.5]}

``imgi`` is percentile-normalized float32 (~[0,1] with tails beyond), so:
  * intensity scale is a plain per-image multiply (shared across channels),
  * gamma is applied on a per-image min/max-normalized copy then mapped back
    (safe for the out-of-[0,1] tails),
  * gauss_noise sigma is relative to the per-image std,
  * cutout fills squares with the per-image mean (~background post-normalize),
  * gauss_blur is a per-channel separable Gaussian.

Each op is sampled independently per image, so a batch sees distinct
realizations. An empty/None config yields ``make(...) -> None`` (no-op).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_SECTIONS = ("intensity", "gauss_blur", "gauss_noise", "cutout")


# ── per-image photometric ops (operate on a (C, H, W) float32 array) ─────────

def _apply_intensity(img, cfg, rng):
    if rng.random() >= cfg.get("p", 1.0):
        return img
    lo, hi = cfg.get("scale", [1.0, 1.0])
    img = img * rng.uniform(lo, hi)
    glo, ghi = cfg.get("gamma", [1.0, 1.0])
    if (glo, ghi) != (1.0, 1.0):
        g = rng.uniform(glo, ghi)
        mn, mx = float(img.min()), float(img.max())
        if mx > mn:
            t = (img - mn) / (mx - mn)
            img = np.clip(t, 0, 1) ** g * (mx - mn) + mn
    return img


def _apply_gauss_noise(img, cfg, rng):
    if rng.random() >= cfg.get("p", 1.0):
        return img
    sigma = cfg.get("sigma_frac", 0.05) * float(img.std())
    if sigma > 0:
        img = img + rng.normal(0.0, sigma, size=img.shape).astype(img.dtype)
    return img


def _apply_cutout(img, cfg, rng):
    if rng.random() >= cfg.get("p", 1.0):
        return img
    C, H, W = img.shape
    n_lo, n_hi = cfg.get("n", [1, 1])
    s_lo, s_hi = cfg.get("size_frac", [0.05, 0.15])
    fill_mode = cfg.get("fill", "mean")
    fill = {"mean": float(img.mean()), "min": float(img.min()), "zero": 0.0}.get(fill_mode, 0.0)
    img = img.copy()
    for _ in range(int(rng.integers(n_lo, n_hi + 1))):
        sh = max(1, int(rng.uniform(s_lo, s_hi) * H))
        sw = max(1, int(rng.uniform(s_lo, s_hi) * W))
        y0 = int(rng.integers(0, max(1, H - sh + 1)))
        x0 = int(rng.integers(0, max(1, W - sw + 1)))
        img[:, y0:y0 + sh, x0:x0 + sw] = fill
    return img


def _apply_gauss_blur(img, cfg, rng):
    if rng.random() >= cfg.get("p", 1.0):
        return img
    from scipy.ndimage import gaussian_filter
    lo, hi = cfg.get("sigma", [0.5, 1.5])
    sigma = rng.uniform(lo, hi)
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        out[c] = gaussian_filter(img[c], sigma=sigma)
    return out


# Order: intensity -> blur -> noise -> cutout (occlusion last so noise doesn't
# repopulate cut squares; blur before noise so noise isn't smoothed away).
_PIPELINE = [
    ("intensity",   _apply_intensity),
    ("gauss_blur",  _apply_gauss_blur),
    ("gauss_noise", _apply_gauss_noise),
    ("cutout",      _apply_cutout),
]


def make(augment_cfg):
    """Build the ``img_transform`` callable for ``train_seg`` from a config dict.

    ``augment_cfg`` is the parsed ``augment:`` mapping from trainer.yaml (or
    None / {}). Returns a callable ``(imgi) -> imgi`` applying the enabled
    photometric ops per image, or ``None`` if nothing is enabled (so
    ``train_seg`` sees a plain ``img_transform=None``).
    """
    cfg = dict(augment_cfg or {})
    enabled = [(name, fn) for name, fn in _PIPELINE if cfg.get(name)]
    if not enabled:
        logger.info("aug_online: no photometric augmentation enabled — img_transform=None")
        return None

    rng = np.random.default_rng(cfg.get("seed", 0))
    logger.info("aug_online: online augmentation enabled — %s", [n for n, _ in enabled])

    def _transform(imgi):
        out = np.array(imgi, dtype=np.float32, copy=True)
        for b in range(out.shape[0]):
            img = out[b]
            for name, fn in enabled:
                img = fn(img, cfg[name], rng)
            out[b] = img
        return out

    return _transform
