"""
Per-cell marker quantification — signal inside Cellpose-SAM masks, for a
channel *other* than the one the cells were segmented on.

The question: given hair-cell masks segmented on the MYO7A channel, how much
HA-tag / eGFP reporter does each cell carry? The answer has to survive two
properties of this data (both measured, see notes/marker_quant_design.md §1):

  * background is large — on DAPI it is two-thirds of the in-mask mean, so a
    raw mean is close to meaningless without local background subtraction;
  * the reporter channel saturates — 8-bit at acquisition, with cells at 99.8%
    saturated pixels, so `mean` is *censored* at the top of its range and
    `frac_saturated` is a feature rather than a diagnostic.

So this module emits a wide per-cell feature table (~55 columns, one row per
cell) rather than a single number, and leaves the positive/negative call to a
later step that reads those rows.

Channel identity is never guessed
---------------------------------
Channels arrive as separate RGB-wrapped sibling TIFs (`<base>_ch00_SV.tif`, …),
each carrying its data in exactly one RGB plane. Neither the channel→dye map nor
the index of the segmented channel is stable across this dataset — `neonate/`
alone mixes two acquisition protocols, one of which has no eGFP at all. So the
caller *chooses* the channel and this module reports, in every row, exactly what
was measured: file, plane, dye, LUT, detector and where that metadata came from.
`discover_channels` supplies the evidence for that choice; it never makes it.

Usage (inside the cellpose container; /data = /media/DATA/Chris/cellpose2D)
--------------------------------------------------------------------------
  # what channels/dyes exist across a folder?  read-only, chooses nothing
  python /helpers/quant/marker_quant.py inventory --dir /data/cellpose_cc/adult

  # what does each channel of one image look like inside vs outside the masks?
  python /helpers/quant/marker_quant.py inspect --seg <seg.npy>

  # measure one channel of one image
  python /helpers/quant/marker_quant.py measure --seg <seg.npy> --channel ch01

  # concatenate everything already measured into one table
  python /helpers/quant/marker_quant.py collect --dir <dir> --out table.csv
"""

import argparse
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
from scipy import ndimage as ndi

# Shared seg/sidecar I/O lives with the IHC/OHC stack; reuse it rather than
# duplicating (project convention). Import is soft so this module still runs
# standalone — cell type and reject flags simply come back empty.
_IHC_DIR = "/helpers/ihc_ohc"
if os.path.isdir(_IHC_DIR) and _IHC_DIR not in sys.path:
    sys.path.insert(0, _IHC_DIR)
else:
    _local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ihc_ohc")
    _local = os.path.normpath(_local)
    if os.path.isdir(_local) and _local not in sys.path:
        sys.path.insert(0, _local)

try:
    from ihc_ohc_crops import load_pred, load_reject  # type: ignore
    _HAVE_IHC = True
except Exception:  # noqa: BLE001
    _HAVE_IHC = False

SCHEMA_VERSION = 1

# Column order of the exported table. Append-only: a model trained on an
# earlier CSV must keep loading later ones (design log §3).
COLUMNS = [
    # provenance — what was measured, and what we know about it
    "schema_version", "image", "seg_file", "cell_id", "channel", "channel_file",
    "plane", "dye", "lut", "detector", "metadata_source", "is_seg_channel",
    # cell context, from the IHC/OHC sidecar
    "cell_type", "cell_type_source", "cell_type_prob", "rejected",
    # geometry
    "area_px", "centroid_y", "centroid_x", "equiv_diam_px", "eccentricity",
    "solidity",
    # raw signal inside the mask
    "mean", "std", "median", "mad", "min", "max",
    "p10", "p25", "p75", "p90", "p99", "sum",
    # measurement quality
    "frac_saturated", "frac_zero",
    # local background
    "bg_median", "bg_mean", "bg_std", "bg_px", "bg_source",
    # background-corrected
    "mean_bgcorr", "sum_bgcorr", "snr", "ratio_bg",
    "frac_above_bg2", "frac_above_bg3",
    # image-level, repeated per row so each CSV stands alone
    "img_bg_median", "img_p50", "img_p99", "img_max", "sat_level",
    "median_cell_diam_px", "n_cells", "n_rejected",
]

# Defaults for the annulus background. 8 px is ~1/3 of a typical 63x hair-cell
# diameter — wide enough to average out shot noise, narrow enough to stay local.
# min_ring_px guards the dense-packing case where the annulus is almost entirely
# other cells; below it the row falls back to the image background and says so.
DEFAULTS = {"ring_px": 8, "min_ring_px": 50}


# ── channel discovery ──────────────────────────────────────────────────────────

# '<base>_ch<NN>' with an optional '_SV' suffix. Both spellings exist in this
# dataset ('_ch02_SV.tif' in three dirs, '_ch02.tif' in neonate(2)). The greedy
# prefix makes the LAST _chNN win, which is right for stems that contain one
# earlier (e.g. '…_8363 20x_20x…_Processed001_ch02_SV').
_CH_RE = re.compile(r"^(?P<base>.*)_ch(?P<ch>\d+)(?P<suf>_SV)?$")


def split_stem(path):
    """('<base>', 'NN', '_SV'|'') for a channel-tagged path, else None.

    Accepts an image path, a seg path or a bare stem — '_seg'/'_pred'/'_quant'
    tails and the extension are stripped first.
    """
    stem = os.path.basename(path)
    for ext in (".npy", ".tif", ".tiff", ".png"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    for tail in ("_seg", "_pred", "_quant"):
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
    m = _CH_RE.match(stem)
    if not m:
        return None
    return m.group("base"), m.group("ch"), (m.group("suf") or "")


def find_metadata(base, *dirs):
    """Path of the Leica MetaData xml for `base`, searching each dir's
    'MetaData/' subfolder then the dir itself. None if absent."""
    for d in dirs:
        if not d:
            continue
        for cand in (os.path.join(d, "MetaData", base + ".xml"),
                     os.path.join(d, base + ".xml")):
            if os.path.isfile(cand):
                return cand
    return None


def read_channel_metadata(xml_path):
    """Per-channel acquisition info from a Leica LAS X export, in channel order.

    Returns [{lut, dye, detector, resolution, vmax}, …] — index i describes
    'ch<i>'. Empty list if the file can't be parsed; callers degrade to
    'dye unknown' rather than failing, since 7 neonate images have no metadata
    at all.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for c in root.findall(".//ImageDescription/Channels/ChannelDescription"):
        info = {"lut": c.get("LUTName") or "", "dye": "", "detector": "",
                "resolution": c.get("Resolution") or "", "vmax": c.get("Max") or ""}
        for prop in c.findall("ChannelProperty"):
            key = prop.findtext("Key")
            val = prop.findtext("Value") or ""
            if key == "DyeName":
                # 'Leica/EGFP' → 'EGFP'; the vendor prefix is noise.
                info["dye"] = val.split("/")[-1].strip()
            elif key == "DetectorName":
                info["detector"] = val.strip()
        out.append(info)
    return out


def image_for_path(path):
    """The image file `path` refers to, or None.

    A `_seg.npy` / `_pred.npy` / `_quant.npy` sidecar resolves to its paired
    image; an image path is returned unchanged. Needed because the CLI is
    driven by seg paths while the GUI is driven by image paths, and composite
    channel discovery has to read actual pixels either way.
    """
    if path.lower().endswith((".tif", ".tiff", ".png")):
        return path
    stem = path
    for tail in ("_seg.npy", "_pred.npy", "_quant.npy"):
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
            break
    else:
        stem = os.path.splitext(stem)[0]
    for ext in (".tif", ".tiff", ".png"):
        if os.path.isfile(stem + ext):
            return stem + ext
    return None


_RGB_NAMES = ("Red", "Green", "Blue")


def discover_in_file_channels(image_path, max_planes=16):
    """Planes of a multi-channel image, offered as individual channels.

    For composite acquisitions exported as ONE file (an RGB stack, or an
    `_overlay.tif`) there are no `_chNN` siblings — the channels are planes of
    the image itself. Every plane is listed rather than guessed between: plane
    order carries no reliable marker identity, exactly as channel index doesn't
    (design log §D1).

    Planes are read straight from the file, so this also sees planes past the
    third — which the GUI's own `imread_2D` truncates away.

    Returns [] for a single-plane image (nothing to choose between).
    """
    try:
        img = np.asarray(tifffile.imread(image_path))
    except Exception:  # noqa: BLE001
        return []
    if img.ndim != 3:
        return []
    if img.shape[0] <= 4 and img.shape[2] > 4:   # (C,H,W) → (H,W,C)
        img = np.moveaxis(img, 0, -1)
    n = int(img.shape[-1])
    if n < 2:
        return []

    out = []
    for i in range(min(n, max_planes)):
        pl = img[..., i]
        # A near-empty plane is padding, or the burned-in scale bar of an
        # RGB-wrapped single channel. Flagged, never hidden — the user decides.
        empty = float(pl.mean()) < 1.0 or float((pl > 0).mean()) < 0.01
        out.append({
            "tag": f"plane{i}",
            "index": i,
            "path": image_path,
            "plane_index": i,        # forced — never auto-detect for these
            "plane": i,
            "dye": "",
            "lut": _RGB_NAMES[i] if n == 3 and i < 3 else "",
            "detector": "",
            "vmax": "",
            "metadata_source": "",
            "is_seg_channel": False,  # cpsam segments on all planes at once
            "in_file": True,
            "empty": empty,
            "exists": True,
        })
    return out


def discover_channels(image_path, source_root=None, max_channels=8):
    """Channels available for `image_path`, annotated with what's known.

    Two layouts exist in this data; both are handled.

    * **split** — one RGB-wrapped file per channel (`<base>_ch00_SV.tif`, …).
      Siblings are looked for in `source_root` (None = the image's own dir).
      Roughly a third of the dataset has only the segmented channel staged, so
      a short result is normal — the caller must surface it rather than
      substituting the segmented channel.
    * **composite** — one file holding every channel as a plane. Used when the
      path carries no `_chNN` token, which is what identifies the split layout.
      Also covers `_overlay.tif`.

    Each entry: tag, index, path, plane_index, dye, lut, detector,
    metadata_source, is_seg_channel, exists.
    """
    parts = split_stem(image_path)
    if parts is None:
        img = image_for_path(image_path)
        return discover_in_file_channels(img) if img else []
    base, seg_ch, suf = parts
    img_dir = os.path.dirname(os.path.abspath(image_path))
    root = source_root or img_dir

    xml_path = find_metadata(base, root, img_dir)
    meta = read_channel_metadata(xml_path) if xml_path else []
    meta_src = os.path.basename(xml_path) if xml_path else ""

    out = []
    for i in range(max_channels):
        tag = f"ch{i:02d}"
        # Try the suffix spelling the seg used first, then the other one — a
        # folder is consistent in practice but this costs one stat call.
        cands = [os.path.join(root, f"{base}_{tag}{s}.tif")
                 for s in ({suf, "", "_SV"} if suf != "_SV" else {"_SV", ""})]
        path = next((p for p in cands if os.path.isfile(p)), None)
        if path is None:
            continue
        info = meta[i] if i < len(meta) else {}
        out.append({
            "tag": tag,
            "index": i,
            "path": path,
            "plane": None,
            "dye": info.get("dye", ""),
            "lut": info.get("lut", ""),
            "detector": info.get("detector", ""),
            "vmax": info.get("vmax", ""),
            "metadata_source": meta_src,
            "is_seg_channel": (f"{i:02d}" == seg_ch),
            "exists": True,
        })
    return out


def channel_label(ch):
    """One-line human label for a channel — the dropdown's whole job is to make
    a wrong pick obvious, so say what's known and admit what isn't."""
    known = [x for x in (ch.get("dye"), ch.get("lut"), ch.get("detector")) if x]
    if known:
        desc = " · ".join(known)
    elif ch.get("in_file"):
        # A plane of a composite: no acquisition metadata to name it by, and
        # plane order implies no marker identity.
        desc = "unnamed channel"
    else:
        desc = "dye unknown (no MetaData)"
    label = f"{ch['tag']} — {desc}"
    if ch.get("in_file"):
        label += " · in this file"
    if ch.get("empty"):
        label += "  (looks empty)"
    if ch.get("is_seg_channel"):
        label += "  (segmented)"
    return label


def channel_from_file(path, seg_path=None):
    """Build a channel entry for an explicitly-picked file (the escape hatch
    when siblings aren't staged). Identity is unknown unless the filename
    happens to carry a _chNN token that metadata can resolve."""
    parts = split_stem(path)
    tag = f"ch{parts[1]}" if parts else os.path.splitext(os.path.basename(path))[0]
    entry = {"tag": tag, "index": int(parts[1]) if parts else -1, "path": path,
             "plane": None, "dye": "", "lut": "", "detector": "", "vmax": "",
             "metadata_source": "", "is_seg_channel": False, "exists": True}
    if parts:
        d = os.path.dirname(os.path.abspath(path))
        xml_path = find_metadata(parts[0], d)
        meta = read_channel_metadata(xml_path) if xml_path else []
        if entry["index"] < len(meta):
            info = meta[entry["index"]]
            entry.update(dye=info["dye"], lut=info["lut"],
                         detector=info["detector"], vmax=info["vmax"],
                         metadata_source=os.path.basename(xml_path))
    if seg_path:
        seg_parts = split_stem(seg_path)
        if seg_parts and parts:
            entry["is_seg_channel"] = seg_parts[1] == parts[1]
    return entry


# ── image loading ──────────────────────────────────────────────────────────────

def load_channel_plane(path, plane=None):
    """(plane float32, plane_index, dtype_max) for one channel image.

    `plane` selects a specific plane and is what composite images use — there
    the planes ARE the channels, so the choice is the caller's and must not be
    second-guessed.

    With `plane=None` the data plane is auto-detected by max sum, which is the
    right behaviour for an RGB-wrapped single channel (`_chNN`): the data sits
    in exactly one plane and which one varies with the dye order. Detection is
    by sum, never by "which plane has nonzero pixels" — the nominally-blank
    planes are NOT empty, they carry a burned-in scale bar (312 px at 72–255 on
    the probe image).
    """
    img = np.asarray(tifffile.imread(path))
    dtype_max = float(np.iinfo(img.dtype).max) if np.issubdtype(
        img.dtype, np.integer) else float(img.max())
    if img.ndim == 2:
        return img.astype(np.float32), None, dtype_max
    if img.ndim == 3:
        # (C,H,W) → (H,W,C) when the small axis leads.
        if img.shape[0] <= 4 and img.shape[2] > 4:
            img = np.moveaxis(img, 0, -1)
        if plane is not None:
            idx = int(plane)
            if not 0 <= idx < img.shape[-1]:
                raise ValueError(f"plane {idx} out of range for {img.shape} "
                                 f"in {os.path.basename(path)}")
        else:
            sums = [float(img[..., c].astype(np.float64).sum())
                    for c in range(img.shape[-1])]
            idx = int(np.argmax(sums))
        return img[..., idx].astype(np.float32), idx, dtype_max
    raise ValueError(f"unexpected image shape {img.shape} in {path}")


# ── cell context (type / reject), from the IHC/OHC sidecar ─────────────────────

_SOURCE_PRIORITY = ("class_map_user", "class_map_fused",
                    "class_map_geom", "class_map_pred")
_PROB_FOR = {"class_map_fused": "class_prob_fused",
             "class_map_geom": "class_prob_geom",
             "class_map_pred": "class_prob"}


def cell_context(seg_path, seg=None):
    """({cid: (class, source, prob)}, rejected_ids, reject_applied).

    Mirrors celltype.display_label_map's precedence — user corrections beat the
    model — so the CSV agrees with what the GUI shows. Degrades to empty when
    the IHC/OHC helpers aren't importable.
    """
    if not _HAVE_IHC:
        return {}, set(), False
    try:
        pred = load_pred(seg_path, seg)
        rej, applied = load_reject(seg_path, seg)
    except Exception:  # noqa: BLE001
        return {}, set(), False
    ctx = {}
    for key in reversed(_SOURCE_PRIORITY):          # low priority first
        cmap = pred.get(key) or {}
        if not cmap:
            continue
        probs = pred.get(_PROB_FOR.get(key, "")) or {}
        tag = key.replace("class_map_", "")
        for cid, cls in cmap.items():
            cid = int(cid)
            p = probs.get(cid, probs.get(str(cid)))
            ctx[cid] = (str(cls), tag,
                        float(p) if isinstance(p, (int, float)) else "")
    return ctx, set(int(c) for c in rej), bool(applied)


# ── measurement ────────────────────────────────────────────────────────────────

def _ring_mask(masks, cid, sl, ring_px):
    """Boolean annulus around cell `cid`, excluding every other mask.

    Works in a padded window around the cell's bounding box and uses an exact
    Euclidean distance transform, so the band is `ring_px` wide in real pixels
    regardless of cell shape.
    """
    r0 = max(sl[0].start - ring_px - 1, 0)
    r1 = min(sl[0].stop + ring_px + 1, masks.shape[0])
    c0 = max(sl[1].start - ring_px - 1, 0)
    c1 = min(sl[1].stop + ring_px + 1, masks.shape[1])
    win = masks[r0:r1, c0:c1]
    cell = win == cid
    # distance from the cell body, measured outside it
    dist = ndi.distance_transform_edt(~cell)
    return (dist > 0) & (dist <= ring_px) & (win == 0), (r0, r1, c0, c1)


def check_pairing(seg_path, channel_path):
    """('ok'|'mismatch'|'unknown', message) — does this channel file belong
    with this seg?

    Shape agreement is NOT a sufficient guard: every image in this dataset is
    1024x1024, so a file from a completely different animal passes it and
    yields plausible numbers against the wrong masks. Base stems, by contrast,
    are globally unique (99 distinct, zero collisions across all four data
    dirs), so a stem mismatch is definitive evidence of a different
    acquisition. 'unknown' means one side carries no _chNN token to compare —
    honest uncertainty, allowed but reported.
    """
    # A composite measured against its own masks is trivially paired — the
    # "channel" is a plane of the very image the seg belongs to.
    if seg_path.endswith("_seg.npy"):
        base_img = seg_path[: -len("_seg.npy")]
        for ext in (".tif", ".tiff", ".png"):
            if os.path.abspath(channel_path) == os.path.abspath(base_img + ext):
                return "ok", ""

    a = split_stem(seg_path)
    b = split_stem(channel_path)
    if a is None or b is None:
        return "unknown", ("cannot verify pairing — no '_chNN' token in "
                           f"{os.path.basename(channel_path)}")
    if a[0] != b[0]:
        return "mismatch", (
            f"'{os.path.basename(channel_path)}' belongs to a different "
            f"acquisition:\n    seg base     {a[0]}\n    channel base {b[0]}")
    return "ok", ""


def measure_seg(seg_path, channel, *, ring_px=None, min_ring_px=None, seg=None,
                allow_mismatch=False):
    """Per-cell signal for one channel of one seg. Returns (rows, meta).

    `channel` is a dict from discover_channels/channel_from_file — the caller's
    explicit choice, never inferred here.

    Refuses when the channel file's base stem identifies a different
    acquisition (unless `allow_mismatch`), and when its dimensions disagree
    with the masks.
    """
    ring_px = int(DEFAULTS["ring_px"] if ring_px is None else ring_px)
    min_ring_px = int(DEFAULTS["min_ring_px"] if min_ring_px is None
                      else min_ring_px)

    pairing, pair_msg = check_pairing(seg_path, channel["path"])
    if pairing == "mismatch" and not allow_mismatch:
        raise ValueError(pair_msg)

    if seg is None:
        seg = np.load(seg_path, allow_pickle=True).item()
    masks = np.asarray(seg["masks"]).squeeze()
    if masks.ndim != 2:
        raise ValueError(f"marker quantification is 2D-only; masks.ndim="
                         f"{masks.ndim}")
    masks = masks.astype(np.int32)

    plane, plane_idx, dtype_max = load_channel_plane(
        channel["path"], channel.get("plane_index"))
    if plane.shape != masks.shape:
        raise ValueError(
            f"channel image {plane.shape} does not match masks {masks.shape} "
            f"— {os.path.basename(channel['path'])} is probably from a "
            f"different acquisition")

    # Saturation ceiling: the acquisition metadata's Max when we have it (Leica
    # records 255 for these 8-bit stacks), else the dtype's max.
    sat_level = dtype_max
    try:
        if channel.get("vmax"):
            sat_level = float(channel["vmax"])
    except (TypeError, ValueError):
        pass

    bg_all = plane[masks == 0]
    img_bg_median = float(np.median(bg_all)) if bg_all.size else 0.0
    img_meta = {
        "img_bg_median": img_bg_median,
        "img_p50": float(np.percentile(plane, 50)),
        "img_p99": float(np.percentile(plane, 99)),
        "img_max": float(plane.max()),
        "sat_level": float(sat_level),
    }

    ctx, rejected_ids, reject_applied = cell_context(seg_path, seg)

    # Geometry in one pass. regionprops needs a positive-int label image and
    # returns one row per present label.
    from skimage.measure import regionprops_table
    props = regionprops_table(
        masks, properties=("label", "area", "centroid", "equivalent_diameter",
                           "eccentricity", "solidity"))
    geom = {int(l): dict(
        area_px=float(props["area"][i]),
        centroid_y=float(props["centroid-0"][i]),
        centroid_x=float(props["centroid-1"][i]),
        equiv_diam_px=float(props["equivalent_diameter"][i]),
        eccentricity=float(props["eccentricity"][i]),
        solidity=float(props["solidity"][i]),
    ) for i, l in enumerate(props["label"])}

    diams = [g["equiv_diam_px"] for g in geom.values()]
    median_diam = float(np.median(diams)) if diams else 0.0

    slices = ndi.find_objects(masks)
    image_stem = os.path.basename(seg_path)
    if image_stem.endswith("_seg.npy"):
        image_stem = image_stem[: -len("_seg.npy")]

    rows = []
    for cid, sl in enumerate(slices, start=1):
        if sl is None or cid not in geom:
            continue
        cell = masks[sl] == cid
        vals = plane[sl][cell]
        if vals.size == 0:
            continue

        ring, (r0, r1, c0, c1) = _ring_mask(masks, cid, sl, ring_px)
        ring_vals = plane[r0:r1, c0:c1][ring]
        if ring_vals.size >= min_ring_px:
            bg_median = float(np.median(ring_vals))
            bg_mean = float(ring_vals.mean())
            bg_std = float(ring_vals.std())
            bg_px = int(ring_vals.size)
            bg_source = "ring"
        else:
            # Dense packing — the annulus is almost all other cells. Say so in
            # the row rather than reporting a background from 6 pixels.
            bg_median = img_bg_median
            bg_mean = float(bg_all.mean()) if bg_all.size else 0.0
            bg_std = float(bg_all.std()) if bg_all.size else 0.0
            bg_px = int(ring_vals.size)
            bg_source = "image"

        mean = float(vals.mean())
        median = float(np.median(vals))
        p10, p25, p75, p90, p99 = (float(x) for x in np.percentile(
            vals, [10, 25, 75, 90, 99]))
        cls, cls_src, cls_prob = ctx.get(cid, ("", "", ""))
        # Guard the degenerate-background cases explicitly: a zero bg_std would
        # make snr infinite, and a zero bg_median would do the same to ratio_bg.
        snr = float((mean - bg_median) / bg_std) if bg_std > 0 else ""
        ratio_bg = float(mean / bg_median) if bg_median > 0 else ""

        rows.append({
            "schema_version": SCHEMA_VERSION,
            "image": image_stem,
            "seg_file": os.path.basename(seg_path),
            "cell_id": int(cid),
            "channel": channel["tag"],
            "channel_file": os.path.basename(channel["path"]),
            "plane": "" if plane_idx is None else int(plane_idx),
            "dye": channel.get("dye", ""),
            "lut": channel.get("lut", ""),
            "detector": channel.get("detector", ""),
            "metadata_source": channel.get("metadata_source", ""),
            "is_seg_channel": bool(channel.get("is_seg_channel", False)),
            "cell_type": cls,
            "cell_type_source": cls_src,
            "cell_type_prob": cls_prob,
            "rejected": bool(cid in rejected_ids),
            **geom[cid],
            "mean": mean,
            "std": float(vals.std()),
            "median": median,
            "mad": float(np.median(np.abs(vals - median))),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "p10": p10, "p25": p25, "p75": p75, "p90": p90, "p99": p99,
            "sum": float(vals.sum()),
            "frac_saturated": float((vals >= sat_level).mean()),
            "frac_zero": float((vals <= 0).mean()),
            "bg_median": bg_median,
            "bg_mean": bg_mean,
            "bg_std": bg_std,
            "bg_px": bg_px,
            "bg_source": bg_source,
            "mean_bgcorr": mean - bg_median,
            "sum_bgcorr": float((vals - bg_median).sum()),
            "snr": snr,
            "ratio_bg": ratio_bg,
            "frac_above_bg2": float((vals > bg_median + 2 * bg_std).mean())
                              if bg_std > 0 else "",
            "frac_above_bg3": float((vals > bg_median + 3 * bg_std).mean())
                              if bg_std > 0 else "",
            **img_meta,
            "median_cell_diam_px": median_diam,
            "n_cells": int(len(geom)),
            "n_rejected": int(len(rejected_ids) if reject_applied else 0),
        })

    meta = {
        "channel": channel["tag"],
        "channel_file": channel["path"],
        "plane": plane_idx,
        "dye": channel.get("dye", ""),
        "lut": channel.get("lut", ""),
        "detector": channel.get("detector", ""),
        "metadata_source": channel.get("metadata_source", ""),
        "is_seg_channel": bool(channel.get("is_seg_channel", False)),
        "ring_px": ring_px,
        "min_ring_px": min_ring_px,
        "schema_version": SCHEMA_VERSION,
        "reject_applied": reject_applied,
        "pairing": pairing,
        **img_meta,
    }
    return rows, meta


def inspect_channels(seg_path, channels, seg=None):
    """In-mask vs background summary for every candidate channel of one image.

    This is the evidence the human picks on — a reporter channel shows a wide
    in-mask spread with a clean background, an antibody channel used for
    segmentation shows uniformly high in-mask signal. Cheap: one read and a few
    reductions per channel.
    """
    if seg is None:
        seg = np.load(seg_path, allow_pickle=True).item()
    masks = np.asarray(seg["masks"]).squeeze().astype(np.int32)
    out = []
    for ch in channels:
        row = {"tag": ch["tag"], "dye": ch.get("dye", ""),
               "lut": ch.get("lut", ""),
               "is_seg_channel": ch.get("is_seg_channel", False)}
        try:
            plane, plane_idx, dtype_max = load_channel_plane(
                ch["path"], ch.get("plane_index"))
            if plane.shape != masks.shape:
                row["error"] = f"shape {plane.shape} != masks {masks.shape}"
                out.append(row)
                continue
            fg = plane[masks > 0]
            bg = plane[masks == 0]
            ids = np.arange(1, int(masks.max()) + 1)
            per_cell = ndi.mean(plane, masks, ids) if ids.size else np.array([])
            row.update(
                plane=plane_idx,
                in_mask_mean=float(fg.mean()) if fg.size else 0.0,
                bg_mean=float(bg.mean()) if bg.size else 0.0,
                p99=float(np.percentile(plane, 99)),
                frac_saturated=float((fg >= dtype_max).mean()) if fg.size else 0.0,
                cell_p5=float(np.percentile(per_cell, 5)) if per_cell.size else 0.0,
                cell_p95=float(np.percentile(per_cell, 95)) if per_cell.size else 0.0,
            )
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)
        out.append(row)
    return out


# ── persistence ────────────────────────────────────────────────────────────────

def quant_path(seg_path):
    """'<stem>_seg.npy' → '<stem>_quant.npy'.

    Deliberately NOT the _pred.npy sidecar: celltype.run_celltype wipes that
    down to class_map_user + mask_reject on every re-predict, which would
    destroy measurements each time the classifier ran (design log §D5).
    """
    if seg_path.endswith("_seg.npy"):
        return seg_path[: -len("_seg.npy")] + "_quant.npy"
    return os.path.splitext(seg_path)[0] + "_quant.npy"


def csv_path(seg_path, channel_tag):
    """Per-channel CSV name — the tag is in the filename so measuring a second
    channel never clobbers the first, while re-measuring one is idempotent."""
    base = quant_path(seg_path)[: -len("_quant.npy")]
    return f"{base}_quant_{channel_tag}.csv"


def load_quant(seg_path):
    p = quant_path(seg_path)
    if not os.path.exists(p):
        return {}
    try:
        d = np.load(p, allow_pickle=True).item()
    except Exception:  # noqa: BLE001
        return {}
    return d if isinstance(d, dict) else {}


def save_quant(seg_path, channel_tag, rows, meta):
    """Merge one channel's measurement into <stem>_quant.npy (atomic write)."""
    cur = load_quant(seg_path)
    channels = dict(cur.get("channels") or {})
    channels[channel_tag] = {"rows": rows, "meta": meta}
    payload = {"schema_version": SCHEMA_VERSION, "channels": channels}
    p = quant_path(seg_path)
    tmp = p + ".tmp"
    # write through a handle so np.save can't append a second '.npy'
    with open(tmp, "wb") as fh:
        np.save(fh, payload, allow_pickle=True)
    os.replace(tmp, p)
    return p


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


# ── reporting ──────────────────────────────────────────────────────────────────

def summarize(rows, meta, *, exclude_rejected=True):
    """Human-readable summary of one measurement, for stdout and the GUI.

    Leads with provenance — which file, which plane, which dye — because the
    single most damaging failure here is measuring the wrong channel and not
    noticing.
    """
    if not rows:
        return "no cells measured"
    ident = meta.get("dye") or "dye unknown"
    lines = [
        f"channel {meta['channel']} · {ident}"
        f"{' · ' + meta['lut'] if meta.get('lut') else ''}"
        f"{'  [SEGMENTED CHANNEL]' if meta.get('is_seg_channel') else ''}",
        f"  file  {os.path.basename(meta['channel_file'])}"
        f"  plane {meta.get('plane')}"
        f"  meta  {meta.get('metadata_source') or 'NONE'}",
    ]

    use = [r for r in rows if not (exclude_rejected and r["rejected"])]
    n_rej = len(rows) - len(use)
    lines.append(f"  cells {len(use)}"
                 + (f"  (excluded {n_rej} off-band)" if n_rej else ""))

    def q(key, sel):
        v = [r[key] for r in sel if isinstance(r[key], (int, float))]
        return np.percentile(v, [25, 50, 75]) if v else [0, 0, 0]

    by_class = {}
    for r in use:
        by_class.setdefault(r["cell_type"] or "unlabeled", []).append(r)
    for cls, sel in sorted(by_class.items()):
        m = q("mean", sel)
        c = q("mean_bgcorr", sel)
        s = q("snr", sel)
        lines.append(
            f"  {cls:<10s} n={len(sel):<5d} "
            f"mean {m[1]:7.1f} [{m[0]:.1f}–{m[2]:.1f}]  "
            f"bgcorr {c[1]:7.1f} [{c[0]:.1f}–{c[2]:.1f}]  "
            f"snr {s[1]:6.2f}")

    bg_img = sum(1 for r in use if r["bg_source"] == "image")
    lines.append(f"  background  image median {meta['img_bg_median']:.1f}"
                 f"  ·  {bg_img}/{len(use)} cells fell back to image bg")

    # Saturation is not a footnote on this data: the reporter channel has cells
    # at 99.8% saturated pixels, where `mean` is a floor rather than a value.
    heavy = sum(1 for r in use if r["frac_saturated"] > 0.5)
    if heavy:
        lines.append(
            f"  WARNING: {heavy}/{len(use)} cells are >50% saturated at "
            f"{meta['sat_level']:.0f} — mean is censored for those; use "
            f"frac_saturated when interpreting the top of the range")
    if not meta.get("metadata_source"):
        lines.append("  WARNING: no acquisition metadata — channel identity "
                     "is unverified")
    if meta.get("pairing") == "mismatch":
        lines.append("  WARNING: this channel file's stem identifies a "
                     "DIFFERENT acquisition — measured only because the "
                     "mismatch was overridden")
    elif meta.get("pairing") == "unknown":
        lines.append("  WARNING: channel file carries no '_chNN' token, so it "
                     "could not be verified as belonging to this image")
    if meta.get("is_seg_channel"):
        lines.append("  NOTE: this is the channel the cells were segmented on "
                     "(useful as a positive control, not as a marker readout)")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _iter_segs(directory):
    for name in sorted(os.listdir(directory)):
        if name.endswith("_seg.npy"):
            yield os.path.join(directory, name)


def cmd_inventory(args):
    """Report the channel→dye mapping across a folder. Chooses nothing."""
    rows = []
    for seg_path in _iter_segs(args.dir):
        chans = discover_channels(seg_path, args.source_root)
        parts = split_stem(seg_path)
        seg_ch = f"ch{parts[1]}" if parts else "?"
        desc = ", ".join(
            f"{c['tag']}={c['dye'] or '?'}"
            f"{'*' if c['is_seg_channel'] else ''}" for c in chans) or "none"
        rows.append((os.path.basename(seg_path), seg_ch, len(chans), desc))
    print(f"{'image':<58s} {'seg':>5s} {'n':>3s}  channels (dye; * = segmented)")
    print("-" * 110)
    groups = {}
    for name, seg_ch, n, desc in rows:
        print(f"{name[:58]:<58s} {seg_ch:>5s} {n:>3d}  {desc}")
        groups.setdefault((seg_ch, desc), []).append(name)
    print(f"\n{len(rows)} images, {len(groups)} distinct configuration(s):")
    for (seg_ch, desc), names in sorted(groups.items(),
                                        key=lambda kv: -len(kv[1])):
        print(f"  n={len(names):<4d} seg on {seg_ch}  |  {desc}")
    if len(groups) > 1:
        print("\nNOTE: configurations differ within this folder — do not assume "
              "a single channel index means the same dye across all images.")
    return 0


def cmd_inspect(args):
    chans = discover_channels(args.seg, args.source_root)
    if not chans:
        print("no channel files found — this image's siblings are not staged "
              "here; pass --source_root or measure with an explicit --file")
        return 1
    print(f"{'channel':<9s} {'dye':<22s} {'lut':<7s} {'plane':>5s} {'in-mask':>9s} "
          f"{'bg':>8s} {'p99':>6s} {'sat':>6s} {'cell p5':>8s} {'cell p95':>9s}")
    print("-" * 96)
    for r in inspect_channels(args.seg, chans):
        if "error" in r:
            print(f"{r['tag']:<9s} ERROR: {r['error']}")
            continue
        star = "*" if r["is_seg_channel"] else " "
        print(f"{r['tag']:<8s}{star} {(r['dye'] or '?'):<22s} "
              f"{(r['lut'] or '?'):<7s} {str(r['plane']):>5s} "
              f"{r['in_mask_mean']:>9.1f} {r['bg_mean']:>8.1f} "
              f"{r['p99']:>6.0f} {r['frac_saturated']:>6.3f} "
              f"{r['cell_p5']:>8.1f} {r['cell_p95']:>9.1f}")
    print("\n* = the channel the masks were segmented on.  A reporter channel "
          "typically shows a wide cell p5→p95 spread over a clean background.")
    return 0


def cmd_measure(args):
    if args.file:
        channel = channel_from_file(args.file, args.seg)
    else:
        chans = discover_channels(args.seg, args.source_root)
        if not chans:
            print("ERROR: no channel files found next to this seg — pass "
                  "--source_root <dir> or --file <path>")
            return 1
        match = [c for c in chans if c["tag"] == args.channel]
        if not match:
            print(f"ERROR: {args.channel} not found. Available: "
                  + ", ".join(channel_label(c) for c in chans))
            return 1
        channel = match[0]

    pairing, pair_msg = check_pairing(args.seg, channel["path"])
    if pairing == "mismatch" and not args.force:
        print(f"ERROR: {pair_msg}\n  refusing — every image here is 1024x1024, "
              f"so shape agreement proves nothing. Pass --force if this really "
              f"is the right file.")
        return 1
    if pairing == "unknown":
        print(f"WARNING: {pair_msg}")

    rows, meta = measure_seg(args.seg, channel, ring_px=args.ring_px,
                             min_ring_px=args.min_ring_px,
                             allow_mismatch=args.force)
    save_quant(args.seg, channel["tag"], rows, meta)
    out = args.out or csv_path(args.seg, channel["tag"])
    write_csv(out, rows)
    print(summarize(rows, meta))
    print(f"  wrote {out}")
    return 0


def cmd_collect(args):
    """Concatenate images ALREADY measured. Never chooses a channel."""
    all_rows = []
    n_img = 0
    for seg_path in _iter_segs(args.dir):
        data = load_quant(seg_path)
        for tag, block in (data.get("channels") or {}).items():
            if args.channel and tag != args.channel:
                continue
            all_rows.extend(block.get("rows") or [])
            n_img += 1
    if not all_rows:
        print("nothing measured yet in this folder — run `measure` per image "
              "first (channel choice is deliberately manual)")
        return 1
    write_csv(args.out, all_rows)
    chans = sorted({r["channel"] for r in all_rows})
    dyes = sorted({r["dye"] or "?" for r in all_rows})
    print(f"collected {len(all_rows)} cells from {n_img} measurement(s) "
          f"→ {args.out}")
    print(f"  channels: {', '.join(chans)}   dyes: {', '.join(dyes)}")
    if len(dyes) > 1:
        print("  NOTE: more than one dye in this table — check `dye` before "
              "pooling; the same channel index is not the same marker across "
              "all acquisitions.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inventory", help="report channel/dye mapping in a folder")
    p.add_argument("--dir", required=True)
    p.add_argument("--source_root", default=None)
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("inspect", help="per-channel signal summary for one image")
    p.add_argument("--seg", required=True)
    p.add_argument("--source_root", default=None)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("measure", help="quantify one channel of one image")
    p.add_argument("--seg", required=True)
    p.add_argument("--channel", default=None, help="e.g. ch01")
    p.add_argument("--file", default=None, help="explicit channel image path")
    p.add_argument("--source_root", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--ring_px", type=int, default=None)
    p.add_argument("--min_ring_px", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="measure even if the channel file's stem says it "
                        "belongs to a different acquisition")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("collect", help="concatenate already-measured images")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--channel", default=None)
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    if args.cmd == "measure" and not args.channel and not args.file:
        ap.error("measure needs --channel or --file (there is no default: "
                 "channel identity is not consistent across this dataset)")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
