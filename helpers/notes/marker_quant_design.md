# Marker quantification — design choice log

**Created 2026-08-16.** Per-cell signal measurement inside Cellpose-SAM masks,
for a channel *other* than the one the cells were segmented on — e.g. "how much
HA-tag / eGFP reporter is in each hair cell". Project 3, stacked on project 1
(masks) and project 2 (IHC/OHC class).

This file is the **decision record**: what was chosen, why, and what was
rejected. Results and usage live in `helpers/quant/README` header + `RUNBOOK.md`
once the pipeline has been run on real data.

---

## 1. What the data actually is

Established by direct inspection on 2026-08-16, not assumed. Every number below
came from reading the files.

### 1.1 Channels are separate sibling files

Leica LAS X exports one RGB-wrapped TIF per acquired channel:

```
<base>_ch00_SV.tif      data in exactly ONE RGB plane, the rest ~empty
<base>_ch01_SV.tif
<base>_ch02_SV.tif      ← usually (not always) the segmented one
<base>_overlay.tif      lossless RGB composite of the three (verified pixel-exact)
MetaData/<base>.xml     per-channel dye / LUT / detector / bit depth
```

Naming varies: `_ch02_SV.tif` in three directories, `_ch02.tif` in `neonate(2)`.

The "blank" planes are **not** zero — one carries 312 nonzero px at values
72–255, a burned-in scale bar. So the data plane must be found by max sum or
variance, **never** by "which plane has any nonzero pixel".

### 1.2 The channel→dye mapping is not stable, and neither is the seg channel

| directory | dye order (ch00, ch01, ch02) | seg on | n |
|---|---|---|---|
| `adult` | DAPI, **EGFP**, ALEXA 647 | ch02 | 32 |
| `neonate` protocol A | DAPI, ALEXA 647, **ALEXA 555** | **ch01** | 16 |
| `neonate` protocol B | DAPI, **EGFP**, ALEXA 647 | ch02 | 12 |
| `neonate` (no MetaData) | unknown | ch02 | 7 |
| `adult(2)`, `neonate(2)` | single channel only | ch02 | 32 |

Three things follow, and they are the reason this pipeline is shaped the way it
is:

- **Two acquisition protocols are mixed inside `neonate/`.** One has no EGFP at
  all (ALEXA 555 in its place).
- **The segmentation channel moves** between ch01 and ch02. It cannot be
  assumed positionally.
- **The one thing that IS consistent** across every image with metadata: the
  segmented channel is ALEXA 647 (the MYO7A antibody). The *index* moves, the
  dye does not. This is an observation, **not** a rule the code relies on.

"HA is usually green" does not resolve it either: in `adult`, green is EGFP (the
reporter); in `neonate` protocol A, green is ALEXA 555 and the reporter channel
is absent entirely.

### 1.3 Background is large and the channel of interest saturates

One adult image (132 cells), in-mask vs non-mask:

| file | bg mean | in-mask mean | p99 | per-cell frac saturated |
|---|---|---|---|---|
| ch00 (DAPI) | 72.1 | 109.0 | 244 | max 0.03 |
| ch01 (**EGFP**) | 31.3 | 113.5 | 255 | **max 0.998**, mean 0.107 |
| ch02 (ALEXA 647, segmented) | 37.6 | 207.7 | 255 | — |

Per-cell ch01 means run 28 (5th pct) → 245 (95th pct): the reporter signal is
**already visibly bimodal**, which is the encouraging part. But:

- DAPI's background is two-thirds of its in-mask mean. **Raw mean without
  background correction is close to meaningless** on this data.
- The reporter channel has cells at 99.8% saturated pixels. `Resolution="8"` in
  the metadata confirms 8-bit at acquisition — there is no higher-bit-depth
  original to fall back on (same wall the head-pilot hit). **Mean intensity is
  censored above ~250 for genuine positives.** `frac_saturated` is a feature,
  not a diagnostic.

### 1.4 Base stems are globally unique

99 distinct base stems across all four directories, **zero collisions**. This is
what makes stem-based lookup against an arbitrary source directory safe.

---

## 2. Design choices

### D1 — Channel selection is an explicit human act, per image

**Decision.** Nothing is auto-selected. The dropdown starts empty of a choice
and export stays disabled until the user picks. No dye-alias table, no
"HA = green" rule, no positional default.

**Why.** §1.2 — there is no convention to encode. Any rule keyed on index,
colour, or dye breaks across two directories of existing data. A wrong channel
produces a plausible number with no error, which is the worst possible failure
mode for a measurement tool.

**Rejected:** dye-based auto-resolution (`--channel egfp` resolving per image).
Attractive for batch, but it silently picks ALEXA 555 on neonate protocol A. Can
be revisited once `inventory` establishes a convention actually exists.

### D2 — The picker is *informed*, not clever

**Decision.** Each dropdown entry carries what is known:
`ch01 — EGFP · Green · HyD S 2`, degrading explicitly to
`ch01 — dye unknown (no MetaData)`. The segmented channel is tagged
`(segmented)`. A permanent provenance line under the dropdown shows the resolved
file path, detected data plane, dye and metadata source, so what is about to be
measured is never implicit. An **inspect** action reports in-mask mean, local
background, p99 and saturated fraction for every discovered channel.

**Why.** Precision here means giving the human the evidence to choose correctly,
then recording the choice — not guessing on their behalf. Inspect is the same
computation that revealed ch01's bimodality in §1.3, and it runs sub-second on
three 1024² uint8 files.

### D3 — Channel source root + stem matching

**Decision.** Channels are looked up as `<base>_ch<NN>[_SV].tif` in a **channel
source root**, defaulting to the image's own directory and overridable with a
folder browser (persisted for the session). Metadata is looked for at
`<root>/MetaData/<base>.xml` and the image dir. A per-image explicit file
override is the escape hatch.

**Why.** §1.4 makes stem matching unambiguous. A folder-level setting (rather
than per-image file picking) is what makes a working session bearable and is the
seam batch would eventually need.

### D4 — Read the file, never the GUI's `self.stack`

**Decision.** Measurement always `tifffile.imread`s the channel file directly.

**Why.** `io._initialize_images` globally min–max rescales the stack to 0–255
float32, and `transforms.convert_image` truncates >3-channel input to 3. The
GUI's array is a *display* product; quantifying it would silently rescale every
image by its own extremes and make cross-image comparison meaningless.

### D5 — Own sidecar `<stem>_quant.npy`, not `_pred.npy`

**Decision.** Measurements go to a dedicated `<stem>_quant.npy`, structured
`{"channels": {tag: {"rows": [...], "meta": {...}}}}`. CSV is its export.

**Why.** `celltype.run_celltype` wipes `_pred.npy` down to `class_map_user` +
`mask_reject*` on **every** re-predict (celltype.py:220-247). Quant values stored
there would be destroyed each time someone re-ran the classifier. A separate
sidecar also avoids extending `PRED_KEYS` and keeps ownership clean.

### D6 — Local ring background with a documented fallback

**Decision.** Per-cell background = median of an annulus `ring_px` wide around
the mask, excluding **all** other masks. If fewer than `min_ring_px` valid
pixels survive, fall back to the image-level non-mask median and record
`bg_source = "image"` in the row.

**Why.** §1.3 — background is too large to ignore and varies across the field.
But the organ of Corti is densely packed, so the annulus is often mostly *other
cells*; a silent fallback would hide that. The `bg_source` column makes every
row auditable.

### D7 — Measure and call are separated at the module boundary

**Decision.** `measure_seg()` returns feature rows and nothing else. No
threshold, no positive/negative call, no fitted model in v1.

**Why.** Choosing an operating point before seeing the feature distributions
across animals is the mistake this project has repeatedly avoided elsewhere
(threshold tuning, adaptive fusion weights, four backbones — all measured, all
flat). Export the features, look at them, then decide. Positivity calling is a
strict superset that reads the same rows (§4).

### D8 — Batch is `inventory` + `collect`, never "measure a folder by dye"

**Decision.** Two folder-level operations, neither of which chooses a channel:

- `inventory --dir` — read-only. Reports each image's channel → dye → LUT →
  detector mapping and flags heterogeneity. This is the tool that *answers* the
  convention question rather than assuming one.
- `collect --dir` — concatenates images **already measured** into one table by
  walking their sidecars. It only aggregates human-confirmed choices.

**Why.** D1. As images are measured one at a time, confirmed choices accumulate
in sidecars, and `collect` assembles a training table out of decisions actually
made. Batch measurement becomes a small, well-founded addition once a convention
is demonstrated.

### D9 — The manifest carries measurement params only

**Decision.** `helpers/quant/marker_quant.yaml` holds `ring_px`,
`min_ring_px`, `schema_version` — **not** channel identity.

**Why.** The repo's convention is "the yaml is the run record", and positivity
calling will need a manifest for its operating point. Having it exist from v1
means that addition is a new block rather than a new file plus registry plus
menu entry. Encoding a channel convention we have not established would
re-introduce D1's failure mode through the back door.

### D10 — The viewer can show the measured channel, as a toggle

**Decision.** A `show`/`hide` button swaps the viewport to the channel being
quantified, masks and outlines untouched. Greyed out until a channel is
selected. Not a merge and not an overlay — one channel at a time, so intensity
is judged without a colour cast from a second one.

Implementation points that matter:

- The plane is read **from the file**, then given the same global min–max →
  0–255 mapping `io._initialize_images` applies to a loaded image, so the
  display sliders keep meaning. Percentile levels are recomputed only when
  auto-adjust is on; hand-set levels are an explicit choice and are respected.
- Broadcast across all three RGB planes → renders greyscale by default and can
  still be colourised through the existing RGB dropdown.
- The previous stack **and** its saturation levels are backed up, so toggling
  back is exact rather than a re-read.
- Forces the raw view first: the filtered/restored view reads `stack_filtered`,
  where the swap would be invisible.
- Follows the dropdown — changing the selection while showing re-renders rather
  than leaving the previous channel on screen.
- On image change the backup is **dropped, not restored** (the new image is
  already in `self.stack` by the time the reset hook runs).

**The guard that makes this safe.** `compute_segmentation` takes `self.stack`
as its input, so running cpsam while the reporter channel was displayed would
segment the wrong channel and produce masks that look almost plausible. The
segmentation entry point restores the real image first and says so in the log.
This is the same class of error as D5 and the stem-pairing guard: a wrong input
that yields a believable output.

**Rejected:** a merged/blended two-channel view. Colocalisation display is a
different job, and blending makes per-cell intensity harder to judge, which is
the whole reason to look.

### D11 — Rejected cells are flagged, not dropped

**Decision.** Every mask gets a row, carrying `rejected` from the hair-cell
off-band reject sidecar. The printed summary honours Option A (excludes applied
rejects from counts); the CSV keeps everything.

**Why.** Matches existing `active_reject_ids` semantics and loses no data —
downstream can filter, but cannot recover what was never written.

---

## 3. Output contract

`<seg stem>_quant_<channel tag>.csv`, one row per cell. The channel tag is in
the filename so measuring a second channel never clobbers the first, while
re-measuring the same channel is idempotent.

| group | columns |
|---|---|
| provenance | `schema_version`, `image`, `seg_file`, `cell_id`, `channel`, `channel_file`, `plane`, `dye`, `lut`, `detector`, `metadata_source`, `is_seg_channel` |
| cell context | `cell_type`, `cell_type_source`, `cell_type_prob`, `rejected` |
| geometry | `area_px`, `centroid_y`, `centroid_x`, `equiv_diam_px`, `eccentricity`, `solidity` |
| raw in-mask | `mean`, `std`, `median`, `mad`, `min`, `max`, `p10`, `p25`, `p75`, `p90`, `p99`, `sum` |
| quality | `frac_saturated`, `frac_zero` |
| background | `bg_median`, `bg_mean`, `bg_std`, `bg_px`, `bg_source` |
| derived | `mean_bgcorr`, `sum_bgcorr`, `snr`, `ratio_bg`, `frac_above_bg2`, `frac_above_bg3` |
| image-level | `img_bg_median`, `img_p50`, `img_p99`, `img_max`, `sat_level`, `median_cell_diam_px`, `n_cells`, `n_rejected` |

Non-obvious columns and why they exist:

- **`snr`** = (mean − bg_median) / bg_std — the per-cell z-score against local
  background. This, not raw mean, is the natural "is it positive" statistic.
- **`frac_above_bg2` / `frac_above_bg3`** — fraction of cell pixels exceeding
  local background by 2σ / 3σ. Catches cells expressing over only part of their
  area, which a mean washes out.
- **`frac_saturated`** — §1.3. Above ~0.5 the mean is a floor, not a
  measurement.
- **`median_cell_diam_px`** — makes `area_px` comparable if 20× data enters
  (cpsam has no rescale step, so magnification is never implicit).
- **image-level columns are repeated on every row** so each CSV is
  self-contained and cross-image normalisation needs no join.

**Schema policy:** append-only, with `schema_version` on every row. A model
trained on v1 CSVs keeps loading later ones.

---

## 4. Extension path to positivity calling

Four seams, so the follow-on work adds rather than rewrites:

1. **Measure ≠ call.** A future `marker_call.py` consumes rows; the feature
   extractor never needs revisiting.
2. **The sidecar is the interchange.** Calls write `positive` / `positive_prob`
   / `caller` into the *same* `<stem>_quant.npy` under the same channel tag. GUI
   tinting then reads it the way `display_label_map` reads class maps — no new
   plumbing.
3. **The manifest already exists** (D9); calling adds a `call:` block.
4. **One panel, built to grow.** A `[caller ▾] [call] [apply]` row joins the
   existing box rather than spawning a second one.

---

## 5. Known limitations

- **32 of 99 local images cannot be quantified at all.** `adult(2)` and
  `neonate(2)` — including the entire `tomt overexp` batch — are single-channel
  with no `MetaData/`. This is a data-staging gap, not a code gap: the other
  channels must be pulled off the acquisition machine. It caps what a positivity
  model can be trained on today.
- **7 neonate images have no metadata**, so their channels can only be
  identified by eye (via `inspect`).
- **8-bit, saturating.** No headroom at the top of the reporter's range; see
  §1.3.
- **`_overlay.tif` is not offered as a source** in v1. It is a lossless
  composite and could serve where individual channel files are missing, but no
  such case exists locally (every image with an overlay also has all three
  channel files). Worth adding if that changes.
- **2D only**, like the rest of the hair-cell stack.

---

## 6. Acceptance tests

1. **Built-in positive control** — measuring the *segmented* channel must show
   in-mask far above background (207 vs 38 on the probe image). If it doesn't,
   masks and image are misaligned.
2. **Idempotence** — measuring the same channel twice yields an identical CSV.
3. **Refusal** — a seg with no sibling channels must refuse, never fall back to
   measuring the segmentation channel and labelling it as the reporter.
4. **Shape guard** — a channel file whose dimensions differ from the masks is a
   hard error, not a broadcast.
5. **Bimodality** — on `adult/`, ch01 `mean_bgcorr` should be bimodal *per
   animal*, not only when pooled.
