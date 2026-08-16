"""
Plot VOC bounding-box annotations and/or Cellpose masks overlaid on a TIF image.

Usage:
  python plot_boxes.py image.tif image.xml                          # boxes only
  python plot_boxes.py image.tif image.xml --seg image_seg.npy      # boxes + masks
  python plot_boxes.py image.tif --seg image_seg.npy                # masks only
  python plot_boxes.py image.tif image.xml --save                   # save PNG
  python plot_boxes.py image.tif image.xml --channel 2              # pick channel
  python plot_boxes.py image.tif image.xml --all-channels           # one panel per channel
"""

import argparse
import colorsys
import os
import xml.etree.ElementTree as ET

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile


# ── XML parsing ────────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path, classes=None):
    """Return (boxes, class_names) from a VOC-style XML annotation file.

    boxes: list of [xmin, ymin, xmax, ymax] ints
    class_names: parallel list of str
    classes: optional set of str — if provided, only matching objects are kept
    """
    root = ET.parse(xml_path).getroot()
    boxes, class_names = [], []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bndbox  = obj.find("bndbox")
        if name_el is None or bndbox is None:
            continue
        name = (name_el.text or "").strip()
        if classes is not None and name not in classes:
            continue
        coords = {tag: bndbox.find(tag) for tag in ("xmin", "ymin", "xmax", "ymax")}
        if any(el is None for el in coords.values()):
            continue
        box = [int(coords[t].text) for t in ("xmin", "ymin", "xmax", "ymax")]
        w, h = box[2] - box[0], box[3] - box[1]
        if w < 5 or h < 5:
            print(f"  skip degenerate box: {name} {box}")
            continue
        boxes.append(box)
        class_names.append(name)
    return boxes, class_names


# ── seg helpers ────────────────────────────────────────────────────────────────

def load_seg(seg_path):
    """Load a cellpose _seg.npy file and return the dict."""
    return np.load(seg_path, allow_pickle=True)[()]


def masks_to_rgba(masks, alpha=0.4, seed=42):
    """Convert an integer label array (H, W) to an RGBA overlay (H, W, 4).

    Background (0) is fully transparent. Each cell ID gets a distinct colour.
    """
    rng = np.random.default_rng(seed)
    n_labels = int(masks.max())
    hues = rng.permutation(n_labels) / max(n_labels, 1)
    colours = np.zeros((n_labels + 1, 4), dtype=float)  # index 0 = background (transparent)
    for i, h in enumerate(hues, start=1):
        colours[i, :3] = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colours[i, 3]  = alpha
    return colours[masks]  # (H, W, 4)


# ── image helpers ──────────────────────────────────────────────────────────────

def load_channel(tif_path, channel):
    """Load a single channel from a multi-channel TIF, returned as (H, W)."""
    img_stack = tifffile.imread(tif_path)
    if img_stack.ndim == 2:
        if channel != 0:
            raise ValueError(f"Image is 2-D (single channel) but --channel={channel} requested")
        return img_stack
    if img_stack.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D array, got shape {img_stack.shape}")
    if img_stack.shape[2] <= 4 and img_stack.shape[0] > 4:
        img_stack = np.moveaxis(img_stack, -1, 0)
    n_channels = img_stack.shape[0]
    if channel >= n_channels:
        raise ValueError(f"Channel {channel} out of range for stack with {n_channels} channels")
    return img_stack[channel]


def load_all_channels(tif_path):
    """Return (C, H, W) array normalised to channels-first."""
    img_stack = tifffile.imread(tif_path)
    if img_stack.ndim == 2:
        return img_stack[np.newaxis]
    if img_stack.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D array, got shape {img_stack.shape}")
    if img_stack.shape[2] <= 4 and img_stack.shape[0] > 4:
        img_stack = np.moveaxis(img_stack, -1, 0)
    return img_stack


def to_display(ch_img):
    """Percentile-stretch a 2-D array to [0, 1] float for imshow."""
    lo, hi = np.percentile(ch_img, (1, 99))
    if hi <= lo:
        return np.zeros_like(ch_img, dtype=float)
    return np.clip((ch_img.astype(float) - lo) / (hi - lo), 0, 1)


# ── colour palette (for boxes) ────────────────────────────────────────────────

_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]

def _class_colour(name, _cache={}):
    if name not in _cache:
        _cache[name] = _PALETTE[len(_cache) % len(_PALETTE)]
    return _cache[name]


# ── core plotting functions ────────────────────────────────────────────────────

def plot_masks(
    ch_img,
    masks,
    *,
    outlines=None,
    title="",   # pass None to suppress title (used when compositing with plot_boxes_and_masks)
    ax=None,
    mask_alpha=0.4,
    outline_colour="white",
    outline_alpha=0.9,
    fontsize=7,
):
    """Overlay Cellpose instance masks on a single-channel image.

    Parameters
    ----------
    ch_img : 2-D array
        Raw pixel data; will be contrast-stretched for display.
    masks : 2-D int array (H, W)
        Integer label array from a cellpose _seg.npy — 0 is background.
    outlines : 2-D bool array (H, W) or None
        Precomputed outlines from the seg dict. If None, outlines are skipped.
    title, ax, mask_alpha, outline_colour, outline_alpha, fontsize : cosmetic tweaks

    Returns
    -------
    ax : the Axes that was drawn into
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(to_display(ch_img), cmap="gray", interpolation="nearest")

    if masks.max() > 0:
        rgba = masks_to_rgba(masks, alpha=mask_alpha)
        ax.imshow(rgba, interpolation="nearest")

    if outlines is not None and outlines.any():
        outlines = outlines.astype(bool)
        outline_rgba = np.zeros((*outlines.shape, 4), dtype=float)
        outline_rgba[outlines] = (*mcolors.to_rgb(outline_colour), outline_alpha)
        ax.imshow(outline_rgba, interpolation="nearest")

    n_cells = int(masks.max())
    if title is not None:
        ax.set_title(f"{title}  [{n_cells} masks]" if title else f"{n_cells} masks",
                     fontsize=fontsize + 2)
    ax.axis("off")

    if standalone:
        plt.tight_layout()

    return ax


def plot_boxes(
    ch_img,
    boxes,
    class_names,
    *,
    title="",
    ax=None,
    linewidth=1.5,
    alpha=0.85,
    fontsize=7,
):
    """Overlay VOC boxes on a single-channel image.

    Parameters
    ----------
    ch_img : 2-D array
        Raw pixel data (any numeric dtype); will be contrast-stretched for display.
    boxes : list of [xmin, ymin, xmax, ymax]
    class_names : list of str, same length as boxes
    title : str
    ax : matplotlib Axes or None — created if not provided
    linewidth, alpha, fontsize : cosmetic tweaks

    Returns
    -------
    ax : the Axes that was drawn into
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(to_display(ch_img), cmap="gray", interpolation="nearest")

    legend_handles = {}

    for box, name in zip(boxes, class_names):
        xmin, ymin, xmax, ymax = box
        colour = _class_colour(name)
        rect = mpatches.FancyBboxPatch(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            boxstyle="square,pad=0",
            linewidth=linewidth,
            edgecolor=colour,
            facecolor="none",
            alpha=alpha,
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            ymin - 2,
            name,
            color=colour,
            fontsize=fontsize,
            va="bottom",
            clip_on=True,
        )
        if name not in legend_handles:
            legend_handles[name] = mpatches.Patch(edgecolor=colour, facecolor="none",
                                                   linewidth=linewidth, label=name)

    if legend_handles:
        ax.legend(handles=list(legend_handles.values()), loc="upper right",
                  fontsize=fontsize + 1, framealpha=0.6)

    ax.set_title(title, fontsize=fontsize + 2)
    ax.axis("off")

    if standalone:
        plt.tight_layout()

    return ax


def plot_boxes_and_masks(
    ch_img,
    boxes,
    class_names,
    masks,
    *,
    outlines=None,
    title="",
    ax=None,
    mask_alpha=0.35,
    linewidth=1.8,
    fontsize=7,
):
    """Overlay both Cellpose masks and VOC boxes on a single-channel image.

    Masks are rendered first as a translucent fill; boxes are drawn on top.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))

    # title=None suppresses plot_masks' own title so we can set the combined one below
    plot_masks(ch_img, masks, outlines=outlines, title=None, ax=ax, mask_alpha=mask_alpha)  # type: ignore[arg-type]

    # draw boxes on top (skip the redundant imshow call)
    legend_handles = {}
    for box, name in zip(boxes, class_names):
        xmin, ymin, xmax, ymax = box
        colour = _class_colour(name)
        rect = mpatches.FancyBboxPatch(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            boxstyle="square,pad=0",
            linewidth=linewidth,
            edgecolor=colour,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin - 2, name, color=colour, fontsize=fontsize,
                va="bottom", clip_on=True)
        if name not in legend_handles:
            legend_handles[name] = mpatches.Patch(edgecolor=colour, facecolor="none",
                                                   linewidth=linewidth, label=name)

    if legend_handles:
        ax.legend(handles=list(legend_handles.values()), loc="upper right",
                  fontsize=fontsize + 1, framealpha=0.6)

    n_cells = int(masks.max())
    suffix = f"{len(boxes)} boxes  |  {n_cells} masks"
    ax.set_title(f"{title}  |  {suffix}" if title else suffix, fontsize=fontsize + 2)

    if standalone:
        plt.tight_layout()

    return ax


# ── high-level helpers used by __main__ ───────────────────────────────────────

def visualise_single(tif_path, xml_path, channel, seg_path=None, classes=None, save=False):
    ch_img = load_channel(tif_path, channel)
    stem   = os.path.splitext(os.path.basename(tif_path))[0]

    boxes, class_names, masks, outlines = [], [], None, None

    if xml_path:
        boxes, class_names = parse_voc_xml(xml_path, classes)

    if seg_path:
        seg     = load_seg(seg_path)
        masks   = seg["masks"]
        outlines = seg.get("outlines")

    fig, ax = plt.subplots(figsize=(12, 9))
    title = f"{stem}  |  channel {channel}"

    if masks is not None and boxes:
        plot_boxes_and_masks(ch_img, boxes, class_names, masks,
                             outlines=outlines, title=title, ax=ax)
    elif masks is not None:
        plot_masks(ch_img, masks, outlines=outlines, title=title, ax=ax)
    else:
        plot_boxes(ch_img, boxes, class_names, title=title, ax=ax)

    plt.tight_layout()

    if save:
        tag = "_boxes_masks" if (masks is not None and boxes) else ("_masks" if masks is not None else "_boxes")
        out = os.path.join(os.path.dirname(os.path.abspath(tif_path)), f"{stem}_ch{channel}{tag}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


def visualise_all_channels(tif_path, xml_path, seg_path=None, classes=None, save=False):
    stack = load_all_channels(tif_path)
    n_ch  = stack.shape[0]
    stem  = os.path.splitext(os.path.basename(tif_path))[0]

    boxes, class_names, masks, outlines = [], [], None, None

    if xml_path:
        boxes, class_names = parse_voc_xml(xml_path, classes)

    if seg_path:
        seg      = load_seg(seg_path)
        masks    = seg["masks"]
        outlines = seg.get("outlines")

    cols = min(n_ch, 4)
    rows = (n_ch + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for ch in range(n_ch):
        r, c = divmod(ch, cols)
        if masks is not None and boxes:
            plot_boxes_and_masks(stack[ch], boxes, class_names, masks,
                                 outlines=outlines, title=f"channel {ch}", ax=axes[r][c])
        elif masks is not None:
            plot_masks(stack[ch], masks, outlines=outlines,
                       title=f"channel {ch}", ax=axes[r][c])
        else:
            plot_boxes(stack[ch], boxes, class_names,
                       title=f"channel {ch}", ax=axes[r][c])

    for idx in range(n_ch, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(stem, fontsize=11)
    plt.tight_layout()

    if save:
        out = os.path.join(os.path.dirname(os.path.abspath(tif_path)), f"{stem}_all_channels.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot VOC bounding boxes and/or Cellpose masks overlaid on a TIF image"
    )
    parser.add_argument("tif",  help="Path to TIF image")
    parser.add_argument("xml",  nargs="?", default=None,
                        help="Path to VOC XML annotation file (optional if --seg is given)")
    parser.add_argument("--seg", default=None, metavar="SEG_NPY",
                        help="Path to a cellpose _seg.npy file; masks are overlaid under the boxes")
    parser.add_argument("--channel", type=int, default=0,
                        help="0-indexed channel to display (default: 0)")
    parser.add_argument("--all-channels", action="store_true",
                        help="Show every channel in a grid instead of a single channel")
    parser.add_argument("--classes", nargs="+", default=None, metavar="CLASS",
                        help="Only show boxes for these classes (e.g. --classes IHC OHC)")
    parser.add_argument("--save", action="store_true",
                        help="Save PNG next to the TIF instead of opening an interactive window")
    args = parser.parse_args()
    if args.xml is None and args.seg is None:
        parser.error("Provide at least one of: xml or --seg")
    return args


if __name__ == "__main__":
    args = parse_args()
    classes = set(args.classes) if args.classes else None

    if args.all_channels:
        visualise_all_channels(args.tif, args.xml, seg_path=args.seg,
                               classes=classes, save=args.save)
    else:
        visualise_single(args.tif, args.xml, args.channel, seg_path=args.seg,
                         classes=classes, save=args.save)
