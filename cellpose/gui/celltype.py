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
import yaml

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
# present in the sidecar — late-fusion is the canonical deployed label.
_SOURCE_PRIORITY = ("class_map_fused", "class_map_geom", "class_map_pred")


def run_celltype(manifest, seg_path):
    """Predict celltype for every mask in seg_path; write back to the sidecar.

    In-process call to the IHC/OHC helpers (no subprocess). The sidecar
    is wiped first so stale predictions from a previous run that used a
    different mask numbering don't linger.

    Returns
    -------
    class_map : dict[int, str]   {mask_id: class_name}, empty on no model output
    source_key : str             which sidecar key the map came from
    """
    _ensure_helpers_on_path()
    # Lazy imports keep torch/sklearn off the GUI's startup path.
    from ihc_ohc_classifier import predict_seg  # type: ignore
    from ihc_ohc_crops import load_pred, pred_path  # type: ignore

    cnn = manifest.get("cnn_ckpt")
    geom = manifest.get("geom_ckpt")
    fuse = manifest.get("fuse_ckpt")
    if not cnn:
        raise RuntimeError("manifest is missing required field 'cnn_ckpt'")
    if fuse and not geom:
        raise RuntimeError("manifest sets fuse_ckpt but no geom_ckpt")

    # Drop any stale sidecar so mask renumbering between sessions can't
    # leave orphan cids behind.
    pp = pred_path(seg_path)
    if os.path.exists(pp):
        try:
            os.remove(pp)
        except OSError:
            pass

    predict_seg(cnn, seg_path, write=True)
    if geom:
        from ihc_ohc_geom_clf import predict_seg_geom  # type: ignore
        predict_seg_geom(geom, seg_path, fuse_ckpt=fuse, write=True)

    pred = load_pred(seg_path)
    for key in _SOURCE_PRIORITY:
        cmap = pred.get(key)
        if cmap:
            return {int(k): str(v) for k, v in cmap.items()}, key
    return {}, ""
