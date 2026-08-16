# CLC hair-cell pipeline — start here

**This is the single entry point.** Status, how the pipeline fits together, and
the commands you actually need. Everything else in `notes/` is depth you reach
for when this file points you there.

Two stacked projects on the same data (mouse cochlear hair cells, confocal):
**segmentation** finds the cells, **IHC/OHC classification** labels each one
inner or outer. Project 2 consumes project 1's masks.

---

## 1. What is deployed

| | model | trained on | held-out estimate |
|---|---|---|---|
| **Segmentation** | `clc_all_full_deploy` | 67 images / 20 animals | AP@0.5 **0.820** · FN-recovery 72% · FP-suppression 90% |
| **IHC/OHC** | `clc-ihc-ohc-v2` (`ihc_ohc/clc_ihc_ohc_v2.yaml`) | 97 images / 29 animals | fused bal_acc **0.977**; +0.164 acc vs v1 on independent labels |
| **Band guard** | code path, not a checkpoint | — | 20× classification 0.196 → 0.965, 63× unchanged |

Segmentation artifacts live in `/ix1/pcody/cellpose/models` (cluster) and
`/data/cellpose_cc/.cellpose/models` (container). Classifier checkpoints live in
`ihc_ohc/runs/<stamp>_clc2-*` and are gitignored.

**Rollback**: v1 of the classifier (`ihc_ohc/clc_ihc_ohc.yaml`, run
`20260624-003309`) is fully intact — re-select it in the GUI dropdown.

## 2. Current state in one paragraph

Segmentation is **closed for now**: every model-side lever has been measured and
returned nothing (thresholds, augmentation, age-specific models, boundary loss,
four backbones, three separate additions of new images, and fixed-scale input).
Two failure modes are **open and unsolved**: severe hair-cell degeneration
(AP 0.18 where the model returns no cells at all) and 20× magnification, which
needs its own model because cpsam has no rescaling step. The residual missed
cells were measured directly in August 2026: **56% merged into a neighbour, 44%
not detected at all** — so any "split touching cells" fix is capped at 86.5%
recovery. Classification improved this round and was redeployed as v2. The
binding constraint on both projects is now **data, not modelling**: 8-bit images
with no boundary information, and a label pool that is ~75% the model's own
accepted output.

## 3. Pipeline

```
confocal .tif ──► Cellpose-SAM fine-tune ──► <stem>_seg.npy   (masks)
                  [project 1, cluster]              │
                                                    ▼
                              TinyHCNet ⊕ geometric fusion
                              [project 2, container]  │
                                                      ▼
                                          <stem>_pred.npy    (class labels)
                                          seg is never mutated
```

**Where things run.** Training and evaluation: Pitt CRC cluster (`ssh pitt_crc`,
`module use ~/modulefiles; module load cellpose_env`). Inference, labelling and
the GUI: the local `cellpose` podman container (`/helpers` = this folder,
`/data` = `/media/DATA/Chris/cellpose2D`).

**Deploying code changes — this bites every time.** Nothing imports the vendored
`cellpose_git/cellpose/` source; the runtime imports the *installed* package.
After any edit under `cellpose/` or to a trainer script:

```bash
./update_container.sh     # from the repo root
./update_cluster.sh       # also copies the pipeline scripts + yamls to ~/cellpose/
```

### Folder layout

| path | what it is |
|---|---|
| `*.py`, `*.slurm`, `submit*.sh`, `trainer.yaml`, `trainer_deploy.yaml` | **the pipeline** — baseline recipe + production recipe only — the only things `update_cluster.sh` deploys. Kept flat because the cluster expects them flat in `~/cellpose/` |
| `ihc_ohc/` | the IHC/OHC classifier, self-contained (container paths — don't move) |
| `quant/` | **marker quantification** — per-cell signal inside the masks for a channel other than the segmented one (HA/eGFP reporter). GUI panel + CLI |
| `notes/` | the experimental record (see §5) |
| `probes/` | one-off diagnostics and experiment watchers. Evidence, not pipeline — nothing depends on them |
| `experiments/` | **settled experiment arms** — configs for hypotheses tested and found not to help (warm-start, FN-augmentation, boundary loss, fixed scale). Has its own README with each result, so nobody re-runs them |
| `legacy/` | superseded: the Cunningham label-transfer path, offline augmentation, the pre-SLURM trainer |
| `bioimageio/` | model card text for external distribution |

## 4. Runbook

Conventions: `$DATA` = `/ix1/pcody/cellpose/data`, `$MODELS` =
`/ix1/pcody/cellpose/models`. Per-fold models are `<RUN_TAG>_fold<k>`; results
land in `~/cellpose/run_logs/<RUN_TAG>/eval_fold<k>.csv`.

```bash
# ── new batch of labelled images ────────────────────────────────────────────
# 1. stage into a NEW dataset dir (old animals symlinked, new files real) so the
#    dataset the deployed model points at stays frozen
ssh pitt_crc "mkdir -p $DATA/CLC_fullN/{adult,neonate}; \
  for sub in adult neonate; do for f in $DATA/CLC_full3/\$sub/*; do \
    ln -sf \"\$f\" $DATA/CLC_fullN/\$sub/; done; done"
rsync -a --files-from=<list> "/media/DATA/Chris/cellpose2D/cellpose_cc/adult(2)/" \
      pitt_crc:$DATA/CLC_fullN/adult/

# 2. fold manifest — ALWAYS --extend, so every existing animal keeps its fold and
#    the new CV stays paired with the old one. --pin spreads a rare phenotype.
python clc_split.py --root $DATA/CLC_fullN --scheme kfold --n_folds 5 --seed 42 \
   --extend $DATA/CLC_full3/clc_folds_kfold_v4.json [--pin 8483=0] \
   --out $DATA/CLC_fullN/clc_folds_kfold_vN.json

# 3. materialise per-fold dirs, then train (one SLURM array task per fold)
python clc_cv.py materialize --manifest <manifest> --recipe clc_all \
   --out_root runs/clc_cv_fullN
FOLD_ROOT=$HOME/cellpose/runs/clc_cv_fullN/clc_all RUN_TAG=<tag> ./submit_cv.sh clc_all

# 4. did it help?  Re-score the PREVIOUS fold models on the new folds as the
#    control (inference only) and compare paired — never compare CV means across
#    different test sets.  Full procedure: notes/RUNBOOK.md §4c
sbatch probes/run_control_eval3.slurm

# ── train the production model (all images, no held-out fold) ───────────────
scp -q trainer_deploy.yaml pitt_crc:cellpose/
ssh pitt_crc 'cd ~/cellpose && sbatch \
  --export=ALL,CELLPOSE_TRAIN_CONFIG=$HOME/cellpose/trainer_deploy.yaml run_trainer.slurm'
python -m cellpose --add_model /path/to/<model>        # register locally

# ── evaluation ─────────────────────────────────────────────────────────────
python clc_cv.py aggregate --manifest <manifest> --results_dir run_logs/<tag> \
   [--baseline_csv <control.csv>]                      # AP + paired Wilcoxon
MODELS='cpsam,heldout:$MODELS/<tag>_fold{k}' DATA_DIRS=... MANIFEST=... \
   sbatch --export=ALL,MODELS,DATA_DIRS,MANIFEST run_erreval.slurm   # FN/FP/TP

# ── IHC/OHC classifier (container) ─────────────────────────────────────────
podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_crops.py \
   --data_dir /data/cellpose_cc/<dir> --out .../crops.npz --channel 0 --group_mode clc
bash runs/clc2_train_driver.sh          # CNN + geom + fuse, one shared stamp
podman exec cellpose python3 /helpers/ihc_ohc/ihc_ohc_pipeline.py predict --dir <dir>

# ── marker quantification (container) ──────────────────────────────────────
# GUI: "marker quantification" panel — pick the channel, click export.
# CLI mirrors it. There is NO default channel, by design (see below).
Q=/helpers/quant/marker_quant.py
podman exec cellpose python3 $Q inventory --dir /data/cellpose_cc/adult   # what dyes exist?
podman exec cellpose python3 $Q inspect --seg <seg.npy>                   # which one is the reporter?
podman exec cellpose python3 $Q measure --seg <seg.npy> --channel ch01    # → _quant_ch01.csv
podman exec cellpose python3 $Q collect --dir <dir> --out table.csv       # pool what's measured
```

Long jobs: submit via SLURM and watch `squeue -M gpu -u $USER`. There is no
`screen` in the container — use `setsid nohup`.

## 5. Rules that are not obvious (each one cost real time)

- **Group by ANIMAL, never by image.** Tonotopic regions from one cochlea must
  not straddle train/test. `clc_split.parse_animal` is the key — and it exists
  **twice** (here and in `ihc_ohc/ihc_ohc_crops.py`); a new ID style must be
  added to both or the two projects silently group the same data differently.
- **New data → `clc_split.py --extend`.** A fresh split reshuffles every animal
  and the new CV can no longer be compared with the old one.
- **Detect label changes by content, never mtime.** The GUI rewrites `_seg.npy`
  whenever the classifier runs. In the 2026-08 batch, 31 of 32 files looked
  updated and only 6 had real edits. Diff `masks`, `ismanual`, `manual_changes`.
- **Know the noise floor before believing a result.** Two runs of the *same*
  recipe on the *same* data differ by ±0.002 averaged over 98 images but ±0.124
  per image on sparse degeneration images. Get a replicate first.
- **Hold error-eval denominators fixed.** Adding one 1262-cell image moved
  FN-recovery from 69.9% to 55.1% with no change in model behaviour.
- **Report GT-cell counts beside every subset metric.** A 4-cell and a 238-cell
  image weigh the same in a per-image mean.
- **Always set `nimg_per_epoch` explicitly** (750). Cellpose defaults it to the
  file count, which silently undertrains on originals-only data.
- **The yaml is the run record.** Change hyperparameters by editing/adding a
  `trainer_*.yaml`, never by CLI override. Every job freezes its resolved config.
- **New model → new `model_name`**, and an entry in `notes/model_train_log.md`.
  Never overwrite an artifact something points at.
- **Percentile normalisation is affine-invariant.** Pre-scaling an image without
  also disabling cellpose's normalisation is a mathematical no-op — it silently
  reproduces the baseline. `clc_error_eval.py` still hardcodes `normalize=True`.
- **A channel index does not identify a marker.** `neonate/` mixes two
  acquisition protocols: ch01 is ALEXA 647 in one and eGFP in the other, and
  the *segmented* channel is ch01 in one and ch02 in the other. Quantification
  therefore never picks a channel for you — read `MetaData/<base>.xml` (or run
  `marker_quant.py inventory`) and choose per image.
- **Matching image dimensions prove nothing.** Every image here is 1024×1024, so
  a file from a different animal passes a shape check and yields plausible
  numbers against the wrong masks. Base stems are what identify an acquisition
  (99 distinct, zero collisions) — that is what `check_pairing` compares.

## 6. Where the detail lives

| document | use it for |
|---|---|
| **`notes/model_train_log.md`** | **the ledger.** Every trained model: config, data, job ID, command, result. §4a–4d are the 2026-07/08 experiments, including the implementation traps. Add an entry for every new model |
| `notes/RUNBOOK.md` | the exhaustive command set — variants, alternate-backbone arms, and the job-ID table |
| `notes/progress.md` | narrative: what was tried, what won, what is closed |
| `notes/clc_seg_results.md` | full results tables; §5 is the final scoreboard |
| `notes/clc_seg_experiments.md` | historical design doc — methodology, not status |
| `ihc_ohc/ihc_ohc.md` | the classifier: derivation, band guard, evaluation caveats |
| `notes/*.png` | failure-mode figures (the evidence behind the FN claims) |

`../../CLAUDE.md` is the agent-facing index and mirrors this file's conventions.
