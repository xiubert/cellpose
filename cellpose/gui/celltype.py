"""
Cell-type classifier integration for the Cellpose GUI.

A "cell-type classifier" is a YAML manifest pointing at one or more
checkpoints that take Cellpose-SAM masks and assign each instance a
class label (e.g. IHC/OHC). The manifest decouples the GUI from the
specific stack — today's IHC/OHC classifier is a TinyHCNet CNN +
geometric logreg + late-fusion, but anything writing the same per-cell
sidecar keys works.

Manifest format
---------------
    name: cunningham-ihc-ohc            # display name in the dropdown
    cnn_ckpt:  /path/to/best.pt         # required
    geom_ckpt: /path/to/geom_best.pkl   # optional
    fuse_ckpt: /path/to/fuse.pkl        # optional (requires geom_ckpt)
    classes:                            # per-class RGB tint (0–255)
      IHC: [242, 64, 51]
      OHC: [38, 166, 242]

Relative ckpt paths are resolved against the manifest's directory, so a
run dir can ship its own manifest and stay portable.

Registry
--------
Manifest paths are listed (one per line) in
~/.cellpose/gui_celltype_models.txt — mirroring how gui_models.txt
tracks user-trained Cellpose models. The GUI reads this at startup;
add/remove via the Models menu.
"""

import os
import pathlib
import sys

import numpy as np

# PyYAML is required for manifest I/O but not for the rest of the Cellpose
# GUI, so it's imported lazily — `from . import celltype` at GUI startup
# must not crash environments that don't have it installed.

CELLTYPE_DIR = pathlib.Path.home().joinpath(".cellpose")
CELLTYPE_LIST_PATH = os.fspath(CELLTYPE_DIR.joinpath("gui_celltype_models.txt"))

# Container path where the IHC/OHC helpers live (see project CLAUDE.md).
# Added to sys.path lazily on first predict so the GUI process can import
# ihc_ohc_classifier / ihc_ohc_geom_clf without polluting the module
# namespace at import time.
_HELPERS_DIR = "/helpers/ihc_ohc"


def _ensure_helpers_on_path():
    if os.path.isdir(_HELPERS_DIR) and _HELPERS_DIR not in sys.path:
        sys.path.insert(0, _HELPERS_DIR)


# ── registry ────────────────────────────────────────────────────────────────────

def list_celltype_models():
    """Return registered manifest paths (existing files only)."""
    if not os.path.exists(CELLTYPE_LIST_PATH):
        return []
    out = []
    with open(CELLTYPE_LIST_PATH) as fh:
        for line in fh:
            p = line.rstrip()
            if p and os.path.exists(p):
                out.append(p)
    return out


def add_celltype_model(manifest_path):
    CELLTYPE_DIR.mkdir(parents=True, exist_ok=True)
    cur = []
    if os.path.exists(CELLTYPE_LIST_PATH):
        with open(CELLTYPE_LIST_PATH) as fh:
            cur = [line.rstrip() for line in fh if line.strip()]
    if manifest_path in cur:
        return False
    cur.append(manifest_path)
    with open(CELLTYPE_LIST_PATH, "w") as fh:
        for p in cur:
            fh.write(p + "\n")
    return True


def remove_celltype_model(manifest_path):
    if not os.path.exists(CELLTYPE_LIST_PATH):
        return
    with open(CELLTYPE_LIST_PATH) as fh:
        cur = [line.rstrip() for line in fh if line.strip()]
    cur = [p for p in cur if p != manifest_path]
    with open(CELLTYPE_LIST_PATH, "w") as fh:
        for p in cur:
            fh.write(p + "\n")


# ── manifest ────────────────────────────────────────────────────────────────────

def load_manifest(path):
    """Read a manifest yaml; resolve relative ckpt paths against its dir."""
    import yaml  # lazy: only needed when the user actually uses celltype
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    base = os.path.dirname(os.path.abspath(path))
    for k in ("cnn_ckpt", "geom_ckpt", "fuse_ckpt"):
        v = cfg.get(k)
        if v and not os.path.isabs(v):
            cfg[k] = os.path.normpath(os.path.join(base, v))
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    cfg.setdefault("classes", {})
    return cfg


def display_name(path):
    """Best-effort label for the registry combo box (manifest 'name' field)."""
    try:
        return load_manifest(path).get("name") or os.path.basename(path)
    except Exception:  # noqa: BLE001
        return os.path.basename(path)


def class_colors_uint8(manifest):
    """{class_name: np.uint8 [R,G,B]} from manifest['classes']; empty if absent."""
    out = {}
    for name, rgb in (manifest.get("classes") or {}).items():
        try:
            r, g, b = (int(x) for x in rgb)
        except (TypeError, ValueError):
            continue
        out[str(name)] = np.array([r, g, b], dtype=np.uint8)
    return out


# ── prediction ──────────────────────────────────────────────────────────────────

# Priority order for which prediction source to display when several are
# present in the sidecar. User corrections override every model output —
# that's the whole point of the labeling tool. Late-fusion is the
# canonical deployed model label below that.
_SOURCE_PRIORITY = ("class_map_user", "class_map_fused",
                    "class_map_geom", "class_map_pred")


def display_label_map(seg_path):
    """Merge every label source in the sidecar into one display dict.

    User labels win per cell, then fused, then geom, then CNN — so a
    cell with both a model prediction and a user correction shows the
    user's label, while a cell with only a model prediction still
    shows up. This is what the GUI tints with after a predict or
    after registering new user labels.

    Returns
    -------
    merged : dict[int, str]   every labeled cid → class name
    source : str              "+"-joined source tags actually present
                              (e.g. "pred+geom+fused+user")
    """
    _ensure_helpers_on_path()
    from ihc_ohc_crops import load_pred  # type: ignore
    pred = load_pred(seg_path)
    merged = {}
    sources = []
    # Apply in REVERSE priority order — later updates win, so user
    # labels (highest priority) overwrite anything underneath them.
    for key in reversed(_SOURCE_PRIORITY):
        cmap = pred.get(key) or {}
        if not cmap:
            continue
        merged.update({int(k): str(v) for k, v in cmap.items()})
        sources.append(key.replace("class_map_", ""))
    # Return source tags in priority order (user first if present),
    # joined into a single human-readable tag for the GUI log line.
    return merged, "+".join(reversed(sources))


def run_celltype(manifest, seg_path):
    """Predict celltype for every mask in seg_path; write back to the sidecar.

    In-process call to the IHC/OHC helpers (no subprocess). Sidecar
    state machine:
      1. Snapshot any existing class_map_user from the sidecar.
      2. Atomically rewrite the sidecar with just those user labels
         (or remove it if none) — wipes stale model predictions
         keyed to a prior mask numbering, but does so AFTER user
         labels are safely persisted so a predict crash can't lose
         them.
      3. predict_seg / predict_seg_geom merge fresh predictions in
         via update_pred.
      4. display_label_map merges every source into one dict for the
         GUI to tint by, user labels winning per cell.

    Returns
    -------
    class_map : dict[int, str]   merged per-cell labels (user > fused > geom > pred)
    source    : str              source tag (e.g. "fused+user")
    """
    _ensure_helpers_on_path()
    # Lazy imports keep torch/sklearn off the GUI's startup path.
    from ihc_ohc_classifier import predict_seg  # type: ignore
    from ihc_ohc_crops import pred_path  # type: ignore

    cnn = manifest.get("cnn_ckpt")
    geom = manifest.get("geom_ckpt")
    fuse = manifest.get("fuse_ckpt")
    if not cnn:
        raise RuntimeError("manifest is missing required field 'cnn_ckpt'")
    if fuse and not geom:
        raise RuntimeError("manifest sets fuse_ckpt but no geom_ckpt")

    # Snapshot the keys that must survive a re-predict — user labels AND the
    # hair-cell reject state — then atomically replace the sidecar with just
    # those. This clears stale predictions keyed to a prior mask numbering
    # while (a) never losing user work to a predict crash and (b) preserving
    # the reject set so Option-A classification still excludes the off-band
    # masks the reject pass applied before this run.
    pp = pred_path(seg_path)
    user_labels = {}
    keep = {}
    if os.path.exists(pp):
        try:
            cur = np.load(pp, allow_pickle=True).item() or {}
            user_labels = {int(k): str(v)
                           for k, v in (cur.get("class_map_user") or {}).items()}
            reject = {int(k): float(v)
                      for k, v in (cur.get("mask_reject") or {}).items()}
            if reject:
                keep["mask_reject"] = reject
                keep["mask_reject_applied"] = bool(
                    cur.get("mask_reject_applied", False))
        except Exception:  # noqa: BLE001
            user_labels = {}
    if user_labels:
        keep["class_map_user"] = user_labels
    if keep:
        tmp = pp + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, keep, allow_pickle=True)
        os.replace(tmp, pp)
    elif os.path.exists(pp):
        try:
            os.remove(pp)
        except OSError:
            pass

    predict_seg(cnn, seg_path, write=True)
    if geom:
        from ihc_ohc_geom_clf import predict_seg_geom  # type: ignore
        predict_seg_geom(geom, seg_path, fuse_ckpt=fuse, write=True)

    return display_label_map(seg_path)


# ── manual labeling ────────────────────────────────────────────────────────────

def set_user_label(seg_path, cell_id, class_name):
    """Persist one (cell_id → class_name) user label to the sidecar.

    Read-modify-writes `class_map_user` so previously-set user labels in
    the same sidecar are preserved (update_pred merges keys but replaces
    each key's value wholesale). The write is atomic via update_pred's
    tmp-file + os.replace.
    """
    return set_user_labels_bulk(seg_path, [cell_id], class_name)


def set_user_labels_bulk(seg_path, cell_ids, class_name):
    """Apply one class to many cells in a single sidecar write.

    Same atomic merge semantics as `set_user_label`, but one
    read-modify-write covers the whole batch — important for the GUI's
    region-select / click-select bulk-labeling path where naive
    per-cell writes would do N redundant disk round-trips.
    """
    _ensure_helpers_on_path()
    from ihc_ohc_crops import load_pred, update_pred  # type: ignore

    pred = load_pred(seg_path)
    cmu = {int(k): str(v) for k, v in (pred.get("class_map_user") or {}).items()}
    for cid in cell_ids:
        cmu[int(cid)] = str(class_name)
    update_pred(seg_path, class_map_user=cmu)
    return cmu


def get_user_labels(seg_path):
    """Read the sidecar's class_map_user, empty dict if none. No I/O cost
    beyond load_pred (sidecar is small)."""
    _ensure_helpers_on_path()
    from ihc_ohc_crops import load_pred  # type: ignore
    pred = load_pred(seg_path)
    return {int(k): str(v) for k, v in (pred.get("class_map_user") or {}).items()}


# ── hair-cell mask post-processing (off-band reject) ────────────────────────────

# Default reject parameters, locked on the CLC validation (link cells within
# 2.5·D, reject an off-band cluster ≥5·D from the band that is ≤0.4× its size):
# 0.03% false deletions across 67 GT images, 0 real detections lost on 9k+ model
# predictions. A manifest's `hair_cell_postprocess:` block overrides any of these.
_REJECT_DEFAULTS = {"eps": 2.5, "min_gap": 5.0, "max_frac": 0.4, "min_cells": 20}


def postprocess_params(manifest):
    """Reject kwargs for a manifest, or None if it doesn't enable the pass.

    The panel is hair-cell-specific: it activates only for a manifest that
    declares a `hair_cell_postprocess:` block (which may be empty → all
    defaults, or override individual params).
    """
    if "hair_cell_postprocess" not in manifest:
        return None                      # key absent → panel disabled
    block = manifest.get("hair_cell_postprocess")   # may be None (bare key)
    params = dict(_REJECT_DEFAULTS)
    if isinstance(block, dict):
        for k in _REJECT_DEFAULTS:
            if k in block:
                params[k] = block[k]
    return params


def run_hair_cell_postprocess(seg_path, params=None):
    """Flag off-band masks for one seg; write the reject set to the sidecar.

    Computes the connected-component off-band reject (`reject_for_seg`) over
    every mask, persists it as `mask_reject` with `mask_reject_applied` reset
    to False (computing never hides masks — the GUI toggle endorses that
    separately). `params` overrides the locked defaults (`_REJECT_DEFAULTS`);
    None uses them — so this works standalone after segmentation, no cell-type
    manifest required. Returns (reject: dict[int, float], applied: bool).
    """
    _ensure_helpers_on_path()
    from ihc_ohc_geom import reject_for_seg  # type: ignore
    from ihc_ohc_crops import set_reject, load_reject  # type: ignore
    params = dict(_REJECT_DEFAULTS) if params is None else params
    seg = np.load(seg_path, allow_pickle=True).item()
    reject = reject_for_seg(seg, **params)
    set_reject(seg_path, reject)
    return load_reject(seg_path)


def get_reject(seg_path):
    """(reject: dict[int, float], applied: bool) from the sidecar."""
    _ensure_helpers_on_path()
    from ihc_ohc_crops import load_reject  # type: ignore
    return load_reject(seg_path)


def set_reject_applied(seg_path, applied):
    """Persist the apply/disable endorsement (True = masks hidden)."""
    _ensure_helpers_on_path()
    from ihc_ohc_crops import set_reject_applied as _sra  # type: ignore
    return _sra(seg_path, applied)
