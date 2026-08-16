"""
Marker quantification integration for the Cellpose GUI.

Measures per-cell signal inside the current image's masks, for a channel
*other* than the one the cells were segmented on — e.g. how much HA-tag /
eGFP reporter each hair cell carries. The heavy lifting lives in
/helpers/quant/marker_quant.py; this module is the thin GUI-side bridge,
mirroring celltype.py.

The one design point worth restating here, because the GUI is where it is
enforced: **channel identity is never guessed.** Neither the channel→dye map
nor the index of the segmented channel is stable across this dataset, so the
dropdown starts with nothing selected, export is disabled until the user picks,
and every export records exactly what was measured (file, plane, dye, LUT,
detector, metadata source). See helpers/notes/marker_quant_design.md.

Session settings
----------------
The channel source directory is remembered in
~/.cellpose/gui_quant.json — sibling channels are frequently not staged
next to the seg, and re-picking the folder for every image would be
unusable.
"""

import json
import os
import pathlib
import sys

CONFIG_DIR = pathlib.Path.home().joinpath(".cellpose")
SETTINGS_PATH = os.fspath(CONFIG_DIR.joinpath("gui_quant.json"))

# Container path where the quant helpers live (see project CLAUDE.md). Added
# lazily so the GUI still starts on a machine without them.
_HELPERS_DIR = "/helpers/quant"


def _ensure_helpers_on_path():
    if os.path.isdir(_HELPERS_DIR) and _HELPERS_DIR not in sys.path:
        sys.path.insert(0, _HELPERS_DIR)


def _mq():
    """Import the helper module on demand (keeps scipy/skimage off startup)."""
    _ensure_helpers_on_path()
    import marker_quant  # type: ignore
    return marker_quant


def available():
    """True when the quant helpers are importable — the panel greys out
    rather than raising if /helpers isn't mounted."""
    try:
        _mq()
        return True
    except Exception:  # noqa: BLE001
        return False


# ── session settings ───────────────────────────────────────────────────────────

def load_settings():
    try:
        with open(SETTINGS_PATH) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_settings(**kv):
    cur = load_settings()
    cur.update({k: v for k, v in kv.items() if v is not None})
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w") as fh:
            json.dump(cur, fh, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not save quant settings: {e}")
    return cur


def source_root():
    root = load_settings().get("source_root")
    return root if root and os.path.isdir(root) else None


# ── discovery / inspection ─────────────────────────────────────────────────────

def discover(image_path, root=None):
    """Candidate channel files for `image_path`, annotated with dye/LUT/detector.

    Empty list is a normal outcome — roughly a third of this dataset has only
    the segmented channel staged locally. The caller must surface that rather
    than substituting the segmented channel.
    """
    return _mq().discover_channels(image_path, root or source_root())


def channel_from_file(path, seg_path=None):
    return _mq().channel_from_file(path, seg_path)


def channel_label(ch):
    return _mq().channel_label(ch)


def check_pairing(seg_path, channel_path):
    """('ok'|'mismatch'|'unknown', message) — guards against measuring a file
    from a different acquisition. Every image here is 1024x1024, so shape
    agreement proves nothing; base stems are what actually identify an image."""
    return _mq().check_pairing(seg_path, channel_path)


def inspect(seg_path, channels):
    """Per-channel in-mask vs background summary, as printable lines.

    This is the evidence the user picks on: a reporter channel shows a wide
    cell p5→p95 spread over a clean background, while the antibody channel the
    masks came from is uniformly high.
    """
    rows = _mq().inspect_channels(seg_path, channels)
    out = [f"{'ch':<6s} {'dye':<22s} {'lut':<7s} {'plane':>5s} {'in-mask':>9s} "
           f"{'bg':>8s} {'p99':>6s} {'sat':>6s} {'cell p5':>8s} {'cell p95':>9s}"]
    for r in rows:
        if "error" in r:
            out.append(f"{r['tag']:<6s} ERROR: {r['error']}")
            continue
        star = "*" if r["is_seg_channel"] else " "
        out.append(f"{r['tag']:<5s}{star} {(r['dye'] or '?'):<22s} "
                   f"{(r['lut'] or '?'):<7s} {str(r['plane']):>5s} "
                   f"{r['in_mask_mean']:>9.1f} {r['bg_mean']:>8.1f} "
                   f"{r['p99']:>6.0f} {r['frac_saturated']:>6.3f} "
                   f"{r['cell_p5']:>8.1f} {r['cell_p95']:>9.1f}")
    out.append("* = the channel the masks were segmented on.  A reporter "
               "channel typically shows a wide cell p5→p95 spread.")
    return "\n".join(out)


# ── measurement ────────────────────────────────────────────────────────────────

def measure(seg_path, channel, *, ring_px=None, min_ring_px=None,
            allow_mismatch=False):
    """Measure one channel and persist both sidecar and CSV.

    Returns (rows, meta, csv_path). The sidecar is <stem>_quant.npy — NOT the
    _pred.npy the classifier owns, because run_celltype wipes that on every
    re-predict and would destroy measurements.
    """
    mq = _mq()
    rows, meta = mq.measure_seg(seg_path, channel, ring_px=ring_px,
                                min_ring_px=min_ring_px,
                                allow_mismatch=allow_mismatch)
    mq.save_quant(seg_path, channel["tag"], rows, meta)
    out = mq.csv_path(seg_path, channel["tag"])
    mq.write_csv(out, rows)
    return rows, meta, out


def summarize(rows, meta):
    return _mq().summarize(rows, meta)


def channel_plane(path):
    """(plane float32, plane_index) for one channel file — the raw data plane,
    read from the file rather than from the GUI's display stack.

    Used by the view toggle. Each `_chNN` TIF wraps its data in one RGB plane
    and which plane varies with the dye order, so the plane is detected rather
    than assumed.
    """
    plane, idx, _dtype_max = _mq().load_channel_plane(path)
    return plane, idx


def measured_channels(seg_path):
    """Channel tags already measured for this seg (for the status line)."""
    try:
        return sorted((_mq().load_quant(seg_path).get("channels") or {}).keys())
    except Exception:  # noqa: BLE001
        return []
