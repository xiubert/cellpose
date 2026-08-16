# CLC segmentation — model training log

**One entry per trained model artifact: the exact code, config, data and
command that produced it.** `progress.md` says *what we learned*;
`RUNBOOK.md` says *how to run things*; this file is the **provenance record** —
given a model file in `/ix1/pcody/cellpose/models/`, this tells you what made it.

**Rule: any new training run gets an entry here** (model name → config → data →
slurm command → job id → result). A model with no entry is unreproducible.

Conventions: `$MODELS` = `/ix1/pcody/cellpose/models`, `$DATA` =
`/ix1/pcody/cellpose/data`. CV runs produce one artifact per fold,
`<RUN_TAG>_fold<k>`; per-image eval CSVs land in
`~/cellpose/run_logs/<RUN_TAG>/eval_fold<k>.csv`; every job freezes its resolved
config next to its log (`run_logs/cellpose_train_<jobid>.config.yaml` for single
runs, `run_logs/<RUN_TAG>/config_fold<k>.yaml` for CV folds).

---

## 0. How a training run is assembled (the code path)

Nothing trains from the vendored `cellpose_git/cellpose/` source directly — the
runtime imports the **installed** cellpose package, so edits must be pushed
first:

```
repo (local)                      cluster (~/cellpose/ + conda env)
─────────────────────────────     ────────────────────────────────────────────
cellpose_git/cellpose/*.py  ──┐
helpers/trainer_slurm.py      ├─ ./update_cluster.sh ─→ site-packages/cellpose/*.py
helpers/trainer*.yaml         │                        ~/cellpose/trainer_slurm.py
helpers/aug_online.py         │                        ~/cellpose/trainer*.yaml
helpers/run_*.slurm, *.sh   ──┘                        ~/cellpose/aug_online.py
                                                       (new files: scp once)
```

Then, at submit time:

```
sbatch run_trainer.slurm            (single run)      ── or ──  submit_cv.sh → run_clc_cv.slurm  (5-fold array)
  │  reads CELLPOSE_TRAIN_CONFIG (a trainer*.yaml)
  │  stages data.source → $SLURM_SCRATCH/{train,test}   (CV: fold_<k>/{train,test})
  │  freezes the config into run_logs/
  └─ python trainer_slurm.py
        │  train: block  → lr / wd / n_epochs / batch_size / nimg_per_epoch / bsize / boundary_weight
        │  augment: block → aug_online.make(cfg) → img_transform callable
        └─ cellpose.train.train_seg(..., img_transform=…, boundary_weight=…)
             └─ model file → ~/cellpose/models/<name> → copied to $MODELS, removed from $HOME
```

Two fork-only hooks are in play (both no-ops at their defaults, so a stock
cellpose reproduces the same run):
- `train_seg(img_transform=…)` — applies `aug_online.py` (intensity/gamma,
  cutout, gaussian noise, gaussian blur) to each **train** crop after cellpose's
  built-in geometric aug. Labels are never touched, nothing hits disk.
- `train_seg(boundary_weight=α)` in `_loss_fn_seg` — flow-gradient boundary
  upweighting. **α = 0.0 = exact stock loss** (what every deployed model uses).

The **alternate-backbone arms** (§4) bypass both: they run upstream-main
cellpose in a separate module env with `trainer_slurm_stock.py`, which reads the
same `train:` block but ignores `augment:` and `boundary_weight`.

---

## 1. PRODUCTION MODEL — `clc_all_full_deploy`

The model currently in prod (local inference + Cellpose GUI + the
[`bioimageio/`](../bioimageio/README.md) export card).

| | |
|---|---|
| **Artifact** | `/ix1/pcody/cellpose/models/clc_all_full_deploy` (1.22 GB, 2026-06-23) |
| **Job** | `2702929` — A100, elapsed **01:37:03**, COMPLETED. Logs `~/cellpose/run_logs/cellpose_train_2702929.{out,err}` |
| **Frozen config** | `~/cellpose/run_logs/cellpose_train_2702929.config.yaml` — **byte-identical** to the repo copy and to `~/cellpose/trainer_deploy.yaml` (md5 `e06428f9…`, verified 2026-07-31) |
| **Config in repo** | [`helpers/trainer_deploy.yaml`](../trainer_deploy.yaml) |
| **Init** | stock `cpsam` (not warm-started) |
| **Data** | `$DATA/CLC_full_deploy` — all **67** curated CLC images / 20 animals, train/ = test/ |
| **Trainer** | `run_trainer.slurm` → `trainer_slurm.py` → fork `train_seg` (α=0, online aug on) |
| **Recipe** | lr 1e-5 · wd 0.1 · 95 ep · batch 8 · nimg_per_epoch 750 · full online aug |
| **Held-out perf** | none — trains on everything. Its honest estimate is the CV twin `clc_all_full_20260604` (§2): **AP@0.5 0.820, FN-recovery 71.9%, FP-suppression 89.6%, TP-retention 95.2%** |

Recipe and config are **identical** to the `clc_all_full_20260604` CV folds —
only `data.source` differs (all 67 images instead of a 4/5 train split). That is
what makes the CV number a valid estimate for this model.

### Exact run code

```bash
# 1. push the config, (re)build the deploy symlink farm, verify
cd /home/pac/Documents/code/cellpose/cellpose_git/helpers
scp -q trainer_deploy.yaml pitt_crc:cellpose/
timeout 90 ssh pitt_crc '
D=/ix1/pcody/cellpose/data/CLC_full; DEP=/ix1/pcody/cellpose/data/CLC_full_deploy
rm -rf $DEP; mkdir -p $DEP/train $DEP/test
for sub in adult neonate; do
  for f in $D/$sub/*_seg.npy $D/$sub/*.tif; do
    ln -sf "$f" $DEP/train/; ln -sf "$f" $DEP/test/
  done
done
echo "train: seg=$(ls $DEP/train/*_seg.npy 2>/dev/null|wc -l) tif=$(ls $DEP/train/*.tif 2>/dev/null|wc -l)"
echo "test:  seg=$(ls $DEP/test/*_seg.npy 2>/dev/null|wc -l) tif=$(ls $DEP/test/*.tif 2>/dev/null|wc -l)"
echo "deploy config data.source:"; module use ~/modulefiles >/dev/null 2>&1; module load cellpose_env >/dev/null 2>&1
python -c "import yaml;d=yaml.safe_load(open(\"$HOME/cellpose/trainer_deploy.yaml\"));print(d[\"data\"][\"source\"],\"| model:\",d[\"train\"][\"model_name\"])"' 2>&1 | tail -6
# expected: train: seg=67 tif=67 / test: seg=67 tif=67
#           /ix1/pcody/cellpose/data/CLC_full_deploy | model: clc_all_full_deploy

# 2. train (1 h 37 m on an A100)
ssh pitt_crc 'cd ~/cellpose && sbatch \
  --export=ALL,CELLPOSE_TRAIN_CONFIG=$HOME/cellpose/trainer_deploy.yaml run_trainer.slurm'

# 3. register for local inference / GUI
python -m cellpose --add_model /path/to/clc_all_full_deploy
```

`CLC_full_deploy/` is **only symlinks** into `CLC_full/{adult,neonate}` — the
`rm -rf` is safe and the rebuild is how newly labeled images enter training.
Always re-read the printed counts: they are the guard that every image made it
in (67 = 32 adult + 35 neonate).

### Frozen config (verbatim, `trainer_deploy.yaml`)

```yaml
data:
  source:    /ix1/pcody/cellpose/data/CLC_full_deploy
  model_dst: /ix1/pcody/cellpose/models
train:
  weight_decay:    0.1
  learning_rate:   1.0e-5
  n_epochs:        95
  batch_size:      8
  nimg_per_epoch:  750
  boundary_weight: 0.0                  # boundary loss was a dead end; stock loss
  model_name:      clc_all_full_deploy
augment:
  seed: 0
  intensity:   {p: 0.5, scale: [0.75, 1.25], gamma: [0.8, 1.25]}
  cutout:      {p: 0.5, n: [1, 3], size_frac: [0.05, 0.15], fill: mean}
  gauss_noise: {p: 0.3, sigma_frac: 0.05}
  gauss_blur:  {p: 0.2, sigma: [0.5, 1.5]}
```

### Retraining it (when new labeled images land)

**Don't retrain blind — measure first.** §4a is the worked example of this exact
procedure (2026-07 batch: measured, tied, not shipped).

1. Stage the new pairs into a **new** dataset dir (`CLC_full2`, …) rather than
   mutating `CLC_full` — the old dir is what the current model's provenance
   points at. 2. Build the fold manifest with **`clc_split.py --extend <prior
   manifest>`** so every existing animal keeps its fold; the new CV is then
   *paired* with the previous one. 3. Run the two arms (§4a): the existing fold
   models re-scored on the extended folds (control, inference only) vs a fresh
   CV on the bigger set. 4. Only if that shows a real gain: rebuild the deploy
   symlink farm over the new dataset and submit with a **new** `model_name`
   (e.g. `clc_all_full_deploy_<YYYYMMDD>`) — never overwrite the artifact the
   GUI/export/manifest points at. 5. Add an entry to this file either way; a
   measured "no change" is as valuable as a win, and stops the next person
   re-running it.

---

## 2. CV / experiment models (fork env, `cellpose_env`)

All rows: from stock `cpsam` unless noted, `run_clc_cv.slurm` via
`submit_cv.sh <recipe>`, fold dirs materialized by
`clc_cv.py materialize --manifest <clc_folds_*.json> --recipe <r>`, evaluated by
`eval_seg.py` on the held-out fold. `RUN_TAG` names the artifacts.

| RUN_TAG (artifacts `_fold<k>`) | job | config | data / folds | headline result |
|---|---|---|---|---|
| **`clc_all_full_20260604`** | 2700928 | `trainer.yaml` | CLC_full 67 img, grouped 5-fold, age-stratified | **AP@0.5 0.820** · FN-rec 71.9% · FP-supp 89.6% · TP-ret 95.2% — **the control every other arm is paired against; twin of the deploy model** |
| `clc_all_fnaug_20260604` | 2701806 | `trainer_fnaug.yaml` (no cutout / no blur) | same folds | FN-rec 72.1%, AP flat → aug content is not the FN lever |
| `clc_adult_full_20260604` | 2701818 (re-run 2702123) | `trainer.yaml` | adult-only recipe | adult AP@0.5 0.828 / FN 79.2% — **tied** with mixed (p=0.68) |
| `clc_neonate_full_20260604` | 2701823 (re-run 2702124) | `trainer.yaml` | neonate-only recipe | neonate AP@0.5 0.804 / FN 62.9% — **loses** to mixed 0.819 (p=0.046) |
| `clc_all_bndry_20260604` | 2702487 | `trainer_bndry.yaml` (α=5) | same folds | AP@0.5 0.814 · FN-rec 71.1% · FP-supp identical → boundary loss does nothing |
| `clc_all_v2base_20260623` | 2702880 | `trainer.yaml`, `INIT=~/.cellpose/models/cpsam_v2` | same folds | AP@0.5 0.8235 (Δ+0.003, p=0.35) — tied |
| `clc_all_full2_20260731` | 3346056 | `trainer.yaml` | **CLC_full2** 87 img (67 + the 2026-07 batch), `clc_folds_kfold_v2.json` | AP@0.5 0.8177 · FN-rec 69.2% — **tied** with the control (p=0.90); see §4a |
| `clc_all_full2_no20x_20260731` | 3346149 | `trainer.yaml` | **CLC_full2_no20x** 86 img (the 20x acquisition dropped), `clc_folds_kfold_v3.json` | AP@0.5 0.8194 · FN-rec 69.3% — **tied** with both other arms (vs B p=0.89); see §4a |
| `clc_all_fixedscale_20260815` | 3481718 | `trainer_fixedscale.yaml` (**fixed 0-255 input, cellpose normalisation off**) | CLC_full3 98 img, v4 folds | AP@0.5 0.8023 — **tied** with the control (p=0.65); no FN-recovery number (see §4d trap 2) |
| `clc_all_full3_20260814` | 3479313 | `trainer.yaml` | **CLC_full3** 98 img (+2026-08 batch, 6 re-curated labels, 20x excluded), `clc_folds_kfold_v4.json` | AP@0.5 0.7996 · FN-rec 69.7% — **tied** with the control (p=0.96); see §4b |
| `clc_all_full4_20260814` | 3479459 | `trainer.yaml` | **CLC_full4** 99 img (= CLC_full3 + the 20x), `clc_folds_kfold_v5.json` | AP@0.5 0.8017 · FN-rec 69.9% — 20x inert for 63x (p=0.36 vs arm D); see §4c |
| `clc_all_rebal_20260814` | 3479460 | `trainer.yaml` | CLC_full3 98 img, `clc_folds_kfold_v6.json` (**8483 pinned to fold 0** so the degeneration animals split across folds) | AP@0.5 0.8055 · FN-rec **70.2%** (best of any arm, +0.9 pt) — still tied (p=0.80); see §4c |
| `clc_all_20260602_1040` — ⚠ artifacts are named **`clc_all_fold<k>`** (this run predates the `<RUN_TAG>_fold<k>` naming) | 2646810 | `trainer.yaml` | **early 15-img** set, LOAO (8 folds) | AP@0.5 0.834 vs cpsam 0.648 (p=0.004) — the original "fine-tuning works" result |
| `clc_all_warm_20260602_1353` | 2647218 | `trainer_warmstart.yaml`, `INIT=$MODELS/label_xfer_aug_retest` | 15-img LOAO | AP@0.5 0.827 — tied with from-cpsam → warm-start not worth the Cunningham dependency |
| `clc_adult_20260602_1530` (folds 4–7 only) | 2647416 | `trainer.yaml` | 15-img LOAO, adult | data-starved (3–4 animals) — superseded by the full-set re-test |
| `clc_neonate_20260602_1530` (folds 0–3 only) | 2647417 | `trainer.yaml` | 15-img LOAO, neonate | ditto |

(Age-specific recipes only produce artifacts for the folds that hold out an
animal of that age — hence the partial fold ranges.)

Reproduce any fork-env CV arm:

```bash
ssh pitt_crc 'cd ~/cellpose; \
  [CELLPOSE_TRAIN_CONFIG=$HOME/cellpose/<trainer_*.yaml>] \
  FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full/<recipe> RUN_TAG=<tag> \
    ./submit_cv.sh <recipe> [INIT_MODEL_PATH]'
```

## 3. Pre-CLC models (Cunningham / PMID-38653806 label transfer)

| model | config | data | note |
|---|---|---|---|
| `label_xfer_aug_retest` | pre-yaml era | 93 Cunningham train imgs × offline D4 (`augment.py`), 95 ep | the **old** best model; superseded on CLC. Scores 0.411 AP@0.5 on CLC (largely an annotation-style artifact — head-only masks vs full-cell GT) |
| `label_xfer_online_aug` | `trainer.yaml` | 93 originals, online aug, `nimg_per_epoch` default (=93) | 0.735 — undertrained; the run that exposed the `nimg_per_epoch` confound |
| `label_xfer_online_aug_nimg750` | `trainer.yaml` (job 2644707) | 93 originals, online aug, **nimg 750** | 0.787 / AP@0.9 tied with offline D4 → **online aug replaces the offline D4 files** |

## 4. Alternate-backbone arms (upstream env, `cellpose_dino_env`)

The 2026-06-23 backbone comparison. Different env, different trainer, **scripts
live only on the cluster** — if this line of work is ever resumed, pull these
back into the repo first:

`~/cellpose/`: `setup_dino_env.sh`, `setup_v2_env.sh`, `trainer_slurm_stock.py`,
`run_clc_cv_dino.slurm`, `run_erreval_dino.slurm`, `trainer_dino384.yaml`,
`run_phase1_dino.slurm`, `run_phase1_v2.slurm`, `sanity_v2_fork.slurm`,
`finish_dino.sh`, `finish_phase2.sh`.

| RUN_TAG | job | backbone / config | CV AP@0.5 (n=67, paired vs 0.8203 control) |
|---|---|---|---|
| `clc_all_dino_stock_20260623` | 2702957 | `cpdino`, stock trainer, `trainer.yaml` | 0.8190 (Δ−0.0013, p=0.85) · FN-rec 71.2 / FP-supp 89.7 / TP-ret 95.1 |
| `clc_all_v2base_stock_20260623` | 2702958 | `cpsam_v2`, stock trainer | 0.8086 (Δ−0.0117, p=0.12) · FN-rec 69.0 / FP-supp 89.4 / TP-ret 95.0 |
| `clc_all_dino384b4_20260623` | 2703588 (earlier attempts 2703047, 2703099) | `cpdino` @ native **384** crop, batch 4 (`trainer_dino384.yaml`) | 0.8177 (Δ−0.0026, p=0.72) |
| `clc_all_v2base_20260623` | 2702880 | `cpsam_v2` + **fork** trainer/online aug (fork env) | 0.8235 (Δ+0.0032, p=0.35) |

Pooling script: `finish_dino.sh` → `run_logs/dino_summary.txt`; FN-recovery eval
job 2703245. **Verdict: every backbone is statistically tied with cpsam** — no
backbone lever on this data. `clc_all_full_deploy` (cpsam) stays.

---

## 4a. Data-scaling re-test — the 2026-07 batch (+20 images) — NO IMPROVEMENT

**Question:** the lab added 20 newly curated images (`cellpose_cc/adult(2)` +
`neonate(2)`). Does training on them improve the deployed model? And does
excluding the one out-of-domain 20x acquisition change that answer?
**Answer: no to both — statistically tied on every metric, in both variants.
Prod model unchanged.**

### The data (87 = 67 + 20)

| new animal | age | imgs | note |
|---|---|---|---|
| `8363` | adult | 4 | new prep ("P28 HET cageatac 1E11"); **one is a 20x acquisition** (342 cells, ~8× smaller cell area, 0 hand-drawn) — the rest are 63x |
| `4L` | neonate | 3 | **existing** animal, re-imaged (no pixel duplicates of the old 4L files) |
| `AL`, `CL`, `DL` | neonate | 3 each | new animals |
| `BL` | neonate | 4 | new animal; two 8 kHz variants — `eGFP` and `GtGFP555` (new stain) |

Same provenance as the rest of CLC (cpsam-proposed, human-curated: `model_path`
= cpsam, `ismanual` 0–105/img, `manual_changes` 17–247/img), same 1024×1024×3
uint8. Verified: **no pixel-level duplicates** against the existing 67. All 20
carry signal in **channel 0** (the old set is mixed c0/c1/c2 — harmless for
segmentation, which consumes all three planes).

Two things had to change before this could run at all:
- **`clc_split.py` `_ANIMAL_RE`** only matched numeric animal ids (`5042L`,
  `1L`), so `AL/BL/CL/DL` raised `could not parse animal id`. Extended to
  `\d+[LR]?` **or** `[A-Z]{1,2}[LR]`. Regression-checked: the old 67 still parse
  to the identical 20 animals with identical per-animal counts.
- **`clc_split.py --extend <prior manifest>`** (new): keeps every already-assigned
  animal in its original fold and places only the new animals (greedy: biggest
  first → smallest fold). Without it, re-splitting reshuffles all 25 animals and
  the new CV run is no longer comparable to the old one.

### Dataset + manifest

```
/ix1/pcody/cellpose/data/CLC_full2/{adult,neonate}   # 87 imgs: 67 symlinks into
                                                     # CLC_full/ + 20 real new files
/ix1/pcody/cellpose/data/CLC_full2/clc_folds_kfold_v2.json   # 25 animals, 5 folds
```
Fold sizes 18/22/17/15/15. Old animals keep their `clc_folds_kfold.json` folds
exactly; the 3 new `4L` images join fold 1 **because 4L was already in fold 1** —
otherwise the control's fold-1 model would have been scoring an animal it trained on.

```bash
# stage (from the repo, local)
rsync -a --files-from=<list> "…/cellpose_cc/adult(2)/"   pitt_crc:$DATA/CLC_full2/adult/
rsync -a --files-from=<list> "…/cellpose_cc/neonate(2)/" pitt_crc:$DATA/CLC_full2/neonate/
ssh pitt_crc 'D=/ix1/pcody/cellpose/data; for sub in adult neonate; do \
  for f in $D/CLC_full/$sub/*; do ln -sf "$f" $D/CLC_full2/$sub/; done; done'
# manifest, pinned to the previous fold assignment
python clc_split.py --root $DATA/CLC_full2 --scheme kfold --n_folds 5 --seed 42 \
  --extend $DATA/CLC_full/clc_folds_kfold.json --out $DATA/CLC_full2/clc_folds_kfold_v2.json
python clc_cv.py materialize --manifest $DATA/CLC_full2/clc_folds_kfold_v2.json \
  --recipe clc_all --out_root runs/clc_cv_full2
```

### The arms (identical recipe; only the training pool differs)

| | arm A — control | arm B — test | arm C — 20x dropped |
|---|---|---|---|
| models | `clc_all_full_20260604_fold<k>` (reused, no retrain) | `clc_all_full2_20260731_fold<k>` | `clc_all_full2_no20x_20260731_fold<k>` |
| trained on | old 67 minus fold *k* | **all 87** minus fold *k* | **86** (no 20x) minus fold *k* |
| config | `trainer.yaml` (lr 1e-5, wd 0.1, 95 ep, batch 8, nimg 750, full online aug, from cpsam) | identical | identical |
| dataset / manifest | `CLC_full` / `clc_folds_kfold.json` | `CLC_full2` / `…_v2.json` | `CLC_full2_no20x` / `…_v3.json` |
| job | **3346061** (inference only) | **3346056** array 0–4, ~1 h 39 m/fold | **3346149** array 0–4, ~1 h 40 m/fold |

Arm C exists because the 20x acquisition is out of domain (different
magnification, ~8× smaller cells, essentially uncurated) and was arm B's single
worst test image — the hypothesis being that it was contaminating training.
All three arms are scored on the **same 86 non-20x images** so C is comparable.

`nimg_per_epoch` is 750 in both, so both arms take the **same number of gradient
steps** — the only variable is the pool crops are drawn from.

```bash
# arm B
ssh pitt_crc 'cd ~/cellpose; FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full2/clc_all \
  RUN_TAG=clc_all_full2_20260731 ./submit_cv.sh clc_all'
# arm A (re-score the control on the extended folds) — run_control_eval.slurm
# watcher (screen): finish_full2.sh -> run_logs/full2_summary.txt, then erreval

# arm C — same, over the 20x-free dataset (folds pinned again via --extend, so
# all three arms stay paired):
ssh pitt_crc 'D=/ix1/pcody/cellpose/data; cd ~/cellpose; \
  python clc_split.py --root $D/CLC_full2_no20x --scheme kfold --n_folds 5 --seed 42 \
    --extend $D/CLC_full2/clc_folds_kfold_v2.json --out $D/CLC_full2_no20x/clc_folds_kfold_v3.json; \
  python clc_cv.py materialize --manifest $D/CLC_full2_no20x/clc_folds_kfold_v3.json \
    --recipe clc_all --out_root runs/clc_cv_full2_no20x; \
  FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full2_no20x/clc_all \
    RUN_TAG=clc_all_full2_no20x_20260731 ./submit_cv.sh clc_all'
# watcher (screen): finish_no20x.sh -> run_logs/no20x_summary.txt, then a 4-way erreval
```

### Results — flat everywhere

**Harness check first:** arm A's re-scored old images reproduce
`run_logs/clc_all_full_20260604/eval_fold*.csv` at **max|Δ AP@0.5| = 0.000000**.

Arm B vs arm A over all 87 (`run_logs/full2_summary.txt`):

| subset | metric | arm B | arm A | Δ | W/T/L | p |
|---|---|---|---|---|---|---|
| **all 87** | AP@0.5 | 0.8177 | 0.8139 | **+0.0037** | 36/9/42 | **0.90** |
| all 87 | AP@0.75 | 0.5747 | 0.5729 | +0.0017 | 41/5/41 | — |
| all 87 | AP@0.9 | 0.3360 | 0.3277 | +0.0083 | 43/3/41 | — |
| old 67 | AP@0.5 | 0.8206 | 0.8203 | +0.0003 | 25/6/36 | 0.48 |
| new 20 | AP@0.5 | 0.8079 | 0.7926 | +0.0153 | 11/3/6 | 0.38 |

All three arms on the same **86 non-20x** images (`run_logs/no20x_summary.txt`):

| subset | metric | A (67) | B (87) | C (86) |
|---|---|---|---|---|
| all 86 | AP@0.5 | 0.8165 | 0.8222 | 0.8194 |
| all 86 | AP@0.75 | 0.5736 | 0.5771 | 0.5790 |
| all 86 | AP@0.9 | 0.3296 | 0.3381 | 0.3383 |
| old 67 | AP@0.5 | 0.8203 | 0.8206 | 0.8193 |
| new 19 | AP@0.5 | 0.8032 | 0.8280 | 0.8200 |

| paired contrast (all 86) | Δ AP@0.5 | W/T/L | p |
|---|---|---|---|
| B vs A | +0.0057 | 36/9/41 | 0.94 |
| C vs A | +0.0029 | 39/9/38 | 0.64 |
| **C vs B** | **−0.0028** | 39/13/34 | **0.89** |

Curation error-eval, all four models on the same 86 images (job **3346206**;
n FN=3568 FP=6519 TP=8815):

| model | FN-recovery | FP-suppression | TP-retention |
|---|---|---|---|
| cpsam | 26.4% | 0.0% | 99.4% |
| arm A `clc_all_full_20260604` | **69.1%** | 89.4% | 95.4% |
| arm B `clc_all_full2_20260731` | 69.2% | 89.3% | 95.8% |
| arm C `clc_all_full2_no20x_20260731` | 69.3% | 89.5% | 95.6% |

(The earlier 87-image run, job 3346128, gave A 69.1 / B 69.2 with TP-ret
94.2 / 93.9 — the TP-retention shift is just the 20x image's 342 cells leaving
the denominator, which is why every arm was re-scored on 86 rather than reusing
those numbers.)

**Verdict: all three tied.** FN-recovery spans 69.1–69.3% — 0.2 pt over 3568 FN
cells is ~7 cells. This is *not* a repeat of the 15→67 rebuild (flat on AP but
+16 pts on FN-recovery — the reason FN-recovery is checked separately at all);
here **every** metric is flat. → **`clc_all_full_deploy` stays the production
model.**

Worth keeping in mind:
- **The 20x image was a bystander, not a contaminant.** Removing it from
  training changed nothing (C vs B p=0.89) and landed nominally *below* arm B.
  It scores badly *as a test image* (arm A 0.592 → arm B 0.427) because it is
  out of domain — that is the model failing on it, not it poisoning the
  training set. Useful corollary: a stray off-domain image among ~86 is
  harmless to keep, so mixed batches don't need pre-filtering.
- **No regression from the new data** (old-67 Δ = +0.0003 / −0.0010) — the new
  images are compatible with the existing domain and belong in the dataset for
  any future refresh.
- The one recurring signal is on the **new-batch subset**: both retrained arms
  beat the control there (B +0.025 p=0.18; C +0.017, 12/0/7 wins, p=0.087).
  Right direction — new data helping its own domain — but a secondary subset at
  n=19 that does not carry the overall comparison. Not treated as a result.
- The diminishing return is the expected shape: 15→67 images (+4.5×) moved
  FN-recovery 16 pts; 67→87 (+1.3×) moves nothing.
- **20x is a separate training target, not a data-cleaning problem.** Two more
  20x tifs already sit unlabeled in `cellpose_cc/adult(2)`
  (`20x mid base_Merged`, `20x hook_merged`). If 20x matters, label those plus
  more and train a dedicated model — cpsam has no diameter-rescale mechanism,
  so one model cannot straddle both magnifications for free.

Artifacts kept (valid CV runs, just not better ones):
`/ix1/pcody/cellpose/models/clc_all_full2_20260731_fold{0..4}` and
`clc_all_full2_no20x_20260731_fold{0..4}`; per-image CSVs in
`run_logs/clc_all_full2_20260731/`, `run_logs/clc_all_full2_no20x_20260731/`,
`run_logs/clc_all_full_20260604_v2eval/` (control on 87); analyses in
`run_logs/full2_summary.txt` and `run_logs/no20x_summary.txt`.

## 4b. Data-scaling re-test #2 — the 2026-08 batch (98 images) — NO IMPROVEMENT

**Question:** another batch arrived (4 new adult animals from a "myo 7e10 tomt
overexp" prep) plus label corrections. Does it beat `clc_all_full_deploy`?
**Answer: no — tied again (AP@0.5 p=0.96; FN-recovery +0.4 pt). Prod unchanged.**
But this round did expose a real, separate failure mode — see "Degeneration" below.

### What changed since the 2026-07 round (content-diffed, not by mtime)

`_seg.npy` **mtimes are not evidence of label edits.** 31 of the 32 files in
`cellpose_cc/{adult,neonate}(2)` were rewritten on 2026-08-13 (the GUI re-saves
the seg when the IHC/OHC classifier runs — matching `_pred.npy` timestamps), but
diffing every key showed:

| change | files | GT-relevant? |
|---|---|---|
| `colors` / `model_path` / `normalize_params` only (± `outlines`) | 29 | **no** — GUI session state; `outlines` is derived from `masks`, which were byte-identical |
| **`masks` + `ismanual` + `manual_changes`** | **6** | **yes** — real re-curation: 4 of the ORIGINAL 67 (`5165 8khz` 124→120, `5168 8khz` 122→121, `5167 32khz` 137→136, `5044L 16khz` 135→134) + `4L 16khz`, `BL 16khz` |
| seg deleted upstream | 1 | the 20x **apex** image (its tif remains) |
| brand new | 13 | 20x **mid base** (1262 cells) + 4 animals × 3: `8477`, `8478`, `8480`, `8483` |

> **Always diff `masks`/`ismanual`/`manual_changes`, never mtimes** —
> `helpers/notes/` has the digest script pattern. `ismanual` matters even when
> `masks` don't change: it defines the FN/TP split for `clc_error_eval.py`.
> Because 6 images' GT moved, **the previous control CSVs were stale and the
> control had to be fully re-scored** — reusing them would have compared new
> predictions against old ground truth.

### Dataset

`/ix1/pcody/cellpose/data/CLC_full3/{adult,neonate}` — **98 images / 29 animals**
(47 adult / 51 neonate). Built fresh, not incrementally: symlinks to the frozen
`CLC_full` for the 63 untouched originals, real re-copied files for the 6
re-curated ones and all 31 from the `(2)` dirs. **The 20x image is excluded by
decision** (user call: 20x is a separate training target, not a data-cleaning
problem — cpsam has no diameter-rescale step). Verified key-by-key: 0 GT
mismatches against the lab's current labels. Manifest `clc_folds_kfold_v4.json`
(`--extend` from v2 → all 25 prior animals keep their folds; 8480→f2,
8477+8483→f3, 8478→f4). Fold sizes 18/22/20/20/18.

The new prep spans full density (`8478` 108–123 cells; `8480` 96–141, two of them
100% hand-drawn — cpsam failed outright) down to **near-total hair-cell loss**
(`8477` 4–13 cells; `8483` 12–65). A rendered check
(`helpers/notes/` figure workflow) confirmed the sparse labels are **complete** —
the tissue really is devoid of hair cells, not partially annotated.

### Arms

| | arm A′ — control | arm D — test |
|---|---|---|
| models | `clc_all_full_20260604_fold<k>`, **re-scored** against current GT | `clc_all_full3_20260814_fold<k>` |
| trained on | old 67 (old labels) minus fold *k* | **all 98** minus fold *k* |
| job | **3479318** (inference only) | **3479313** array 0–4, ~1 h 40 m/fold |

Plus **`clc_all_full_deploy` scored directly on the 31 images it has never
seen** (`CLC_full3_newonly`, job 3479318) — leak-free, and the most direct read
on what production actually does on new data.

### Results — tied overall

| subset | n | GT cells | arm A′ | arm D | Δ AP@0.5 | W/T/L | p |
|---|---|---|---|---|---|---|---|
| **ALL** | 98 | 13215 | 0.7907 | 0.7996 | +0.0089 | 43/10/45 | **0.96** |
| old 67 | 67 | 9757 | 0.8210 | 0.8169 | −0.0041 | 26/8/33 | 0.33 |
| 2026-07 batch | 19 | 2619 | 0.8024 | 0.8106 | +0.0082 | 10/1/8 | 0.45 |
| 2026-08 batch | 12 | 839 | 0.6031 | 0.6855 | +0.0823 | 7/1/4 | 0.58 |
| — sparse (8477/8483) | 6 | 134 | 0.3889 | 0.5245 | +0.1356 | 3/1/2 | 0.63 |
| **ALL minus sparse** | 92 | 13081 | 0.8169 | 0.8175 | **+0.0006** | 40/9/43 | 0.79 |

Curation error-eval, 98 images (job **3479399**; n FN=4040 FP=7068 TP=9175):

| model | FN-recovery | FP-suppression | TP-retention |
|---|---|---|---|
| cpsam | 29.8% | 0.0% | 99.3% |
| arm A′ `clc_all_full_20260604` | **69.3%** | 89.2% | 95.5% |
| arm D `clc_all_full3_20260814` | 69.7% | 89.0% | 95.5% |

**+0.4 pt of FN-recovery over 4040 FN cells ≈ 16 cells. Tied.**

### Don't be fooled by the sparse subset

The +0.136 on the degeneration images is **small-denominator noise, not learning**:

| image | GT | deploy | arm A′ | arm D | detections A′→D |
|---|---|---|---|---|---|
| 8477 16khz | 4 | 0.000 | 0.000 | **0.000** | 0 → 0 |
| 8477 32khz | 13 | 0.167 | 0.214 | 0.154 ↓ | 4 → 2 |
| 8477 8khz | 5 | 0.167 | 0.167 | 0.800 ↑ | 2 → 4 |
| 8483 8khz | 35 | 1.000 | 0.944 | 0.917 ↓ | 35 → 34 |
| 8483 16khz | 12 | 0.375 | 0.258 | 0.500 ↑ | 27 → 21 |
| 8483 32khz | 65 | 0.881 | 0.750 | 0.776 ↑ | 61 → 70 |

3 better / 3 worse; the whole subset gain is one image picking up **two
detections**. AP on a 4–13 cell image moves in ~8–25% quanta, so per-image
means mixing a 4-cell and a 238-cell image are not comparable — always report
GT-cell counts beside these subsets. The one well-powered per-image gain is
`8480 16khz "lower myo sig"` **+0.245** on 126 cells, cancelled by losses of
similar size on `4L 16khz` (−0.185/148), `DL 8khz` (−0.128/161), `6L 16khz`
(−0.121/238).

### Degeneration phenotype — a REAL gap, and more data of this kind did not fix it

`clc_all_full_deploy` on the 31 never-seen images: mean AP@0.5 **0.744**, median
**0.872**. Split out:

| subset | n | mean AP@0.5 |
|---|---|---|
| severe degeneration (≤13 GT cells) | 4 | **0.177** |
| everything else | 27 | **0.828** |
| — 2026-07 batch | 16 | 0.788 |
| — 2026-08 batch, normal density | 8 | 0.879 |
| — 8363 63x | 3 | 0.907 |

On normal-density new images prod performs **exactly at its certified 0.82** —
including the new TOMT prep and the new `GtGFP555` stain, which it handles fine.
The entire deficit is 4 severely degenerated images, where it returns **zero
detections** (8477 16khz: 4 GT, 0 predicted) or over-detects on damaged tissue
(8483 16khz: 12 GT, 21 predicted). **Retraining on two degeneration animals does
not fix it** — 8477 16khz is still 0/4 in arm D. This needs its own intervention
(many more degeneration cochleae, or a prep-specific lower `cellprob_threshold`),
not another general retrain.

Caveat on the design: the fold balancer put **both** degeneration animals in
fold 3, so fold 3's model trains without any severely-sparse image. It does train
on the same prep (8478, 8480 sit in other folds), so prep-level transfer was
testable — and produced nothing reliable.

Artifacts: `/ix1/pcody/cellpose/models/clc_all_full3_20260814_fold{0..4}`;
`run_logs/clc_all_full3_20260814/`, `run_logs/clc_all_full_20260604_v4eval/`,
`run_logs/deploy_on_newonly.csv`, `run_logs/full3_summary.txt`.

## 4c. Two follow-ups: the 20x image, and a hold-out that can see the sparse case

Run 2026-08-14 after §4b. Neither changes the deployed model. The **noise floor**
measured here (below) is the most reusable result in this file — it sets the bar
under which no subset result on this dataset should be believed.

### Test 1 — arm E: include the 20x image (`CLC_full4`, 99 images, job 3479459)

`CLC_full4` = `CLC_full3` + the 20x mid-base acquisition (1262 cells, 1095
hand-drawn — far better curated than the apex image dropped in §4a). Manifest
`clc_folds_kfold_v5.json` (`--extend` from v4; the 20x belongs to animal 8363 →
fold 3).

**What this design can and cannot answer:** because the 20x sits in fold 3, no
arm ever trains on 20x *and* tests on it. It measures the effect of a 20x image
in the **training pool on the 63x set** — the deploy-relevant question. With one
20x image and leave-one-animal-out, "does training on 20x help 20x" is not
answerable at all.

| comparison | n | Δ AP@0.5 | p |
|---|---|---|---|
| arm E vs arm D (98 shared 63x images) | 98 | +0.0022 | 0.36 |
| arm E vs control | 98 | +0.0110 | 0.57 |

FN-recovery (comparable 98-image basis, job **3479588**): **69.9%** / FP-supp
89.5% / TP-ret 95.7% — in line with every other arm.

> **Denominator trap:** arm E's first error-eval (job 3479562, over all 99) read
> **55.1%** FN-recovery. That is *not* a regression — including the 20x adds its
> 1095 hand-drawn cells to the FN pool (4040 → 5135) and the model recovers
> almost none of them. Re-run on the 98-image basis it is 69.9%. **Always hold
> the FN/FP/TP denominators fixed when comparing error-eval numbers.**

**The 20x image itself is unusable by every model:**

| model | AP@0.5 | predicted / 1262 true |
|---|---|---|
| `clc_all_full_deploy` | 0.126 | 205 |
| control fold-3 | 0.101 | 169 |
| arm E fold-3 | 0.027 | 40 |

Under 16% of cells found. cpsam has no diameter-rescale step, so a 63x-trained
model has no mechanism to adapt to ~3× smaller cells. → **Including the 20x is
harmless but useless. 20x needs its own model (or explicit rescaling at
inference); it is not a data-cleaning problem.**

### Test 2 — arm F: split the degeneration animals across folds (job 3479460)

§4b could not measure whether degeneration data transfers, because the balancer
put **both** degeneration animals (8477, 8483) in fold 3 — the fold that holds
them out trained on neither. Fixed with the new **`clc_split.py --pin
ANIMAL=FOLD`** flag (applied after `--extend`, so every other animal keeps its
historical fold and all prior comparisons stay paired):

```bash
python clc_split.py --root $DATA/CLC_full3 --scheme kfold --n_folds 5 --seed 42 \
  --extend $DATA/CLC_full3/clc_folds_kfold_v4.json --pin 8483=0 \
  --out $DATA/CLC_full3/clc_folds_kfold_v6.json
```
Now fold 3 tests 8477 with 8483 **in training**, and fold 0 tests 8483 with 8477
**in training**. The control needed matching top-ups (job **3479472**): 8483
re-scored by the **fold-0** control model, since using a different control fold
would compare the arms under different held-out conditions.

| contrast | n | Δ AP@0.5 | p |
|---|---|---|---|
| 8477 (the clean transfer test) F vs D | 3 | +0.1043 | 1.0 |
| 8483 F vs D | 3 | −0.0390 | 0.5 |
| all 6 degeneration, F vs D | 6 | +0.0326 | 0.88 |
| non-degeneration 92, F vs D | 92 | +0.0042 | 0.45 |
| ALL 98, F vs control | 98 | +0.0144 | 0.80 |
| 6 degeneration, F vs control | 6 | +0.1611 | 0.19 |

FN-recovery: **70.2%** / 89.2% / 95.6% — nominally the best of any arm (+0.9 pt
over control ≈ 36 cells of 4040).

The eye-catching single result: `8477 32khz` went **2 → 11 detections** (of 13
true), 0.154 → 0.600. Then the noise floor killed it.

### The noise floor — measure this before believing any subset result

Arms D and E have **identical fold-3 training data** (the 20x is held out in fold
3, so adding it to the dataset changes nothing fold 3 sees), and 8477+8483 both
live in fold 3 under v4/v5. So **D vs E on those six images is a same-data
replicate**: one recipe, one dataset, two runs.

| GT cells | arm D | arm E (identical training data) | \|diff\| |
|---|---|---|---|
| 4 | 0.000 | 0.000 | 0.000 |
| 13 | 0.154 | **0.400** | **0.246** |
| 5 | 0.800 | **0.400** | **0.400** |
| 35 | 0.917 | 0.917 | 0.000 |
| 12 | 0.500 | 0.526 | 0.026 |
| 65 | 0.776 | 0.847 | 0.071 |

**Mean run-to-run |ΔAP| = 0.124, max 0.400, with no data change at all.** The
2→11 detection jump is inside that band — arm E moved the same image 0.154 →
0.400 on nothing but a different seed.

| scope | replicate Δ (E − D, same data) |
|---|---|
| 6 degeneration images | −0.0094 (means partly cancel; per-image spread is huge) |
| all 98 images | **+0.0022** |

**Consequences, and they are general:**
- Per-image AP on 4–65 cell images is worth ±0.12 of nothing. Any subset claim
  over a handful of sparse images needs a replicate before it means anything —
  this retroactively explains §4b's "+0.136 sparse gain".
- 98-image means are stable to ~0.002, so the "tied" headline verdicts across
  §4a/§4b/§4c **are** well powered. It is only the small subsets that are mute.
- Cheapest way to get a replicate: two arms that differ only in data held out of
  the fold you care about — or just re-run one arm with a different seed.

**Verdict: no arm ships.** `clc_all_full_deploy` remains production. Degeneration
transfer is unproven (2 animals is too few); 20x is a separate model.

Artifacts: `clc_all_full4_20260814_fold{0..4}`, `clc_all_rebal_20260814_fold{0..4}`;
`run_logs/{clc_all_full4_20260814,clc_all_rebal_20260814}/`,
`run_logs/ef_summary.txt`, `run_logs/ctrl_fold3_on_20x.csv`,
`run_logs/ctrl_fold0_on_8483.csv`, `run_logs/deploy_on_20x.csv`.

## 4d. OPEN LEAD — percentile normalisation is content-dependent (2026-08-15)

**Not yet acted on. Inference-only finding, no retraining, deterministic (same
model + same image ⇒ exact deltas, no noise floor to clear).**

Cellpose normalises each image to its own `[p1, p99]`. That scale therefore
depends on **how many cells happen to be in frame**. On the degeneration images
the cells occupy 0.5–3% of pixels instead of ~15%, and `p99` lands *below* the
median cell pixel:

| GT cells | % frame in cells | p99 | median cell px | % cell px below p99 | deployed AP |
|---|---|---|---|---|---|
| 4 | 0.47% | 105 | **124** | **42.6%** | 0.000 |
| 5 | 0.68% | 81 | **85** | **48.2%** | 0.167 |
| 12 | 1.20% | 61 | 47 | 63.0% | 0.375 |
| dense (n=58) | 15.3% | 239 | 102 | **93.4%** | ~0.82 |

So on the sparsest images >half of every cell's signal sits above the
normalisation ceiling, versus 6.6% in the regime the model was trained on.

**Measured, 6 degeneration images** (`runs/norm_test*.log`):

| preprocessing | mean AP | 12-cell img | 65-cell img |
|---|---|---|---|
| default `[1,99]` | 0.431 | 0.375 (21 pred) | 0.881 |
| percentile `[0.1,99.9]` | 0.478 | **0.769** | **0.701** ← regresses |
| **fixed 0–255, `normalize=False`** | **0.489** | 0.667 | 0.890 |
| no scaling at all (raw 0–255) | 0.000 | — | — (control: model needs ~[0,1]) |

**Paired over all 98 local images** (`runs/norm_full.log`), default vs fixed 0–255:

| subset | n | default | fixed | Δ | W/T/L |
|---|---|---|---|---|---|
| dense ≥80 cells | 92 | 0.8909 | 0.8897 | **−0.0012** | 29/28/35 |
| sparse <80 | 6 | 0.4315 | 0.4890 | **+0.0576** | 3/2/1 |
| ALL | 98 | 0.8628 | 0.8652 | +0.0024 (p=0.75) | 32/30/36 |

**Conclusions:**
- Fixed absolute scaling is **neutral on the 92 dense images** (−0.001) and
  **positive on the sparse subset** (+0.058), i.e. it fixes the over-detection
  without relocating the damage the way a percentile re-tune does.
- **It does NOT explain the zero-detection failures.** The 4- and 5-cell images
  predict 0–3 masks under *all six* preprocessing regimes tested. That failure is
  upstream of normalisation — most likely a training-distribution effect (every
  training image has 100+ cells in rows), not preprocessing.
- The sparse gain is n=6 and driven mainly by one image. Treat as a lead.

**Follow-up 1 — FN partition: DONE (job 3481711), and it corrects the record.**
Each of the 98 images scored by the fold model that held it out
(`helpers/probes/fn_partition.py`):

| | count | share of residual |
|---|---|---|
| hand-drawn FN cells | 4040 | |
| recovered by the model | 2801 (69.3%, matches the error-eval exactly) | |
| **residual FN** | **1239** | |
| — **merged** (centroid inside a predicted mask) | **694** | **56.0%** |
| — **missed outright** (centroid in background) | **545** | **44.0%** |

**The residual FN is NOT simply merging.** `progress.md` and
`clc_seg_results.md` describe it as under-segmentation/merging of dense OHCs;
that is true of 56% of it. The other **44% is tissue where the model predicts
nothing at all**, which no splitter, boundary loss or threshold change can
address — there is no mask there to split.

Consequences:
- It bounds the splitter idea: a *perfect* split-proposer takes FN-recovery
  69.3% → **86.5%** and no further. Perfect detection of the missed half gets
  to 82.8%. Neither alone passes ~87%.
- It re-reads the boundary-loss null (§progress): α=5 could only ever have
  addressed 56% of the residual, so a muted result was over-determined.
- The split is uneven per image (`7L 16k` 17 merged/0 missed; `6L 16khz`
  5 merged/14 missed), so any intervention should be gated on which regime an
  image is in, not applied globally.
**Follow-up 2 — retrain under fixed scaling: DONE (job 3481718), TIED.**
`clc_all_fixedscale_20260815`, `trainer_fixedscale.yaml` (`train.fixed_scale:
255`), same CLC_full3 v4 folds as arm D so it is paired; evaluated with
`eval_seg.py --fixed_scale 255` so training and scoring preprocessing match.

| subset | n | fixed | control (arm D) | Δ | p |
|---|---|---|---|---|---|
| ALL | 98 | 0.8023 | 0.7996 | +0.0027 | 0.65 |
| dense ≥80 cells | 92 | 0.8225 | 0.8175 | +0.0050 | 0.57 |
| sparse <80 | 6 | 0.4929 | 0.5245 | **−0.0315** | 0.88 |

Per-image sparse: the 12-cell image improves (0.500→0.833, 21→10 predictions —
the same over-detection fix seen at inference time), but the 5-cell image drops
0.800→0.333 and the 4-cell stays at 0. **All of that is inside the ±0.124
per-image seed noise measured for this subset**, so the sparse column is not
interpretable without a replicate.

**Conclusion: training under fixed scaling is tied.** The inference-time benefit
(+0.058 sparse, −0.001 dense) did **not** transfer into a training-time benefit.
The two are different interventions and this is not a contradiction — the
inference-only change repairs a test-time mismatch for sparse images, whereas
training under fixed scale removes the mismatch on both sides and lands back at
parity. The dense subset is nominally +0.005 (~2× the 98-image replicate floor
of 0.0022) but not significant.

**So the standing recommendation is unchanged:** the *inference-only* fixed
scaling remains the cheap option (neutral on 92 dense, +0.058 on 6 sparse,
deterministic), and retraining buys nothing on top of it.

### Two implementation traps this arm exposed (both would have produced a
### plausible but meaningless result)
1. **Pre-scaling without disabling normalisation is a NO-OP.** Percentile
   normalisation is affine-invariant: `(x/s − p1/s)/(p99/s − p1/s) ≡
   (x − p1)/(p99 − p1)`. `trainer_slurm.py` therefore passes
   `normalize=False` whenever `train.fixed_scale` is set. Caught before the run.
2. **`clc_error_eval.py` hardcodes `normalize=True`** (line ~93), so it cannot
   score a fixed-scale model correctly — it would apply percentile normalisation
   to a model never trained under it. An error-eval submitted for this arm was
   cancelled for that reason; the arm has **no FN-recovery number**. Add a
   `--fixed_scale` flag there before ever scoring such a model.

If fixed scaling is ever adopted at inference it must be applied **everywhere at
once** (GUI, `ihc_ohc_pipeline predict`, `eval_seg.py`, `clc_error_eval.py`) or
results stop being comparable across the record above.

## 5. Companion evaluation jobs (no model produced)

| job | what | script |
|---|---|---|
| 2646778 | CLC baselines: stock `cpsam` 0.648 vs `label_xfer_aug_retest` 0.411 | `run_eval.slurm` |
| 2699771 | cellprob × flow threshold sweep (over-detect + filter — refuted) | `run_sweep.slurm` |
| 2701730 | full-set curation error-eval (FN-rec / FP-supp / TP-ret) | `run_erreval.slurm` |
| 2703245 | same, for the backbone arms | `run_erreval_dino.slurm` |
| 3346061 | control arm re-scored on the extended 87-image folds (§4a) | `run_control_eval.slurm` |
| 3346128 | curation error-eval, both arms, 87 images (§4a) | `run_erreval.slurm` |
| 3346206 | curation error-eval, all 3 arms, 86 images (§4a) | `run_erreval.slurm` |
| 3479318 | control re-scored on 98 + **deployed model on the 31 unseen new images** (§4b) | `run_control_eval3.slurm` |
| 3479399 | curation error-eval, control + arm D, 98 images (§4b) | `run_erreval.slurm` |
| 3479472 | control top-ups: fold-3 on the 20x, fold-0 on 8483, deploy on the 20x (§4c) | `run_control_topup.slurm` |
| 3479561 / 3479562 | error-eval, arms F / E (§4c) — **E's 99-image run is denominator-shifted, use 3479588** | `run_erreval.slurm` |
| 3479588 | error-eval, arm E on the comparable 98-image basis (§4c) | `run_erreval.slurm` |
