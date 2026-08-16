# CLC segmentation — runbook (exact commands)

End-to-end commands to reproduce the CLC segmentation experiments. All training
/ eval runs on the Pitt CRC cluster (`ssh pitt_crc`); data prep can be local
(container) or on the cluster. Scripts live in `cellpose_git/helpers/`; on the
cluster they are deployed to `~/cellpose/` (see step 0).

Conventions: `$DATA` = `/ix1/pcody/cellpose/data`, `$MODELS` =
`/ix1/pcody/cellpose/models`. Per-run model names are `<RUN_TAG>_fold<k>`.
Outputs land in `~/cellpose/run_logs/<RUN_TAG>/eval_fold*.csv` and `run_logs/*.out`.

**Production model:** `clc_all_full_deploy` — its full run code is §4a below.
Every trained artifact's config / data / job id is logged in
[`model_train_log.md`](model_train_log.md); **add an entry there for any new
model you train.**

---

## 0. Deploy code to the runtimes (after any edit under `helpers/` or `cellpose/`)

```bash
# from the repo root (local machine)
./update_container.sh           # push cellpose/* into the local podman container
./update_cluster.sh             # push cellpose/*.py + trainer scripts to the cluster
                                # (resolves the conda-env cellpose dir; copies trainer_*.py,
                                #  *.yaml, run_*.slurm, submit*.sh, aug_online.py, clc_*.py)
```
`update_cluster.sh` copies the trainer scripts + helpers to `~/cellpose/`. If you
add a NEW script, confirm it's in the copy list or `scp` it once:
`scp helpers/<new>.py pitt_crc:cellpose/`.

---

## 1. Stage a dataset to the cluster (once per dataset)

Full CLC set is local at `/media/DATA/Chris/cellpose2D/cellpose_cc/{adult,neonate}/`
(container: `/data/cellpose_cc/...`). Stage only each `*_seg.npy` + its matching
`<stem>.tif` (skip the ch00/ch01/overlay tifs and `_rot/_flip` aug copies):

```bash
BASE=/media/DATA/Chris/cellpose2D/cellpose_cc; LIST=/tmp/clc_list.txt; : > $LIST
for sub in adult neonate; do
  for seg in "$BASE/$sub"/*_seg.npy; do
    name=$(basename "$seg"); case "$name" in *_rot*|*_flip*) continue;; esac
    stem=${name%_seg.npy}
    echo "$sub/$name" >> $LIST; [ -f "$BASE/$sub/$stem.tif" ] && echo "$sub/$stem.tif" >> $LIST
  done
done
ssh pitt_crc 'mkdir -p /ix1/pcody/cellpose/data/CLC_full/{adult,neonate}'
( cd "$BASE" && rsync -a --files-from=$LIST ./ pitt_crc:/ix1/pcody/cellpose/data/CLC_full/ )
```

---

### 1a. Adding a NEW batch of labeled images

Stage into a **new** dataset dir (old animals as symlinks, new files real) so the
previous dataset — the provenance of the deployed model — stays frozen:

```bash
D=/ix1/pcody/cellpose/data
ssh pitt_crc "mkdir -p $D/CLC_full2/{adult,neonate}; \
  for sub in adult neonate; do for f in $D/CLC_full/\$sub/*; do \
    ln -sf \"\$f\" $D/CLC_full2/\$sub/; done; done"
# then send only <stem>_seg.npy + <stem>.tif pairs of the new batch (names contain
# spaces/parens -> use --files-from, and skip tifs that have no seg):
rsync -a --files-from=/tmp/clc2_adult.txt   "/media/DATA/Chris/cellpose2D/cellpose_cc/adult(2)/"   pitt_crc:$D/CLC_full2/adult/
rsync -a --files-from=/tmp/clc2_neonate.txt "/media/DATA/Chris/cellpose2D/cellpose_cc/neonate(2)/" pitt_crc:$D/CLC_full2/neonate/
```
Check before staging: same `model_path`/`ismanual` provenance, 1024×1024×3 uint8,
**no pixel duplicates** of existing images, and that `clc_split.parse_animal`
resolves every new stem (a new animal-id style needs a regex update — e.g. the
lettered `AL/BL/CL/DL` ids of the 2026-07 batch).

## 2. Build the leak-free split (group by animal)

```bash
ssh pitt_crc 'module use ~/modulefiles; module load cellpose_env; cd ~/cellpose; \
  python clc_split.py --root /ix1/pcody/cellpose/data/CLC_full \
    --scheme kfold --n_folds 5 --seed 42'        # grouped 5-fold, age-stratified
# -> /ix1/.../CLC_full/clc_folds_kfold.json
# (use --scheme loao for leave-one-animal-out; -> clc_folds_loao.json)

# EXTENDING an existing split after new data (keeps every old animal in its old
# fold, places only the new ones -> the new CV run stays PAIRED with the old one)
ssh pitt_crc 'module use ~/modulefiles; module load cellpose_env; cd ~/cellpose; \
  python clc_split.py --root /ix1/pcody/cellpose/data/CLC_full2 \
    --scheme kfold --n_folds 5 --seed 42 \
    --extend /ix1/pcody/cellpose/data/CLC_full/clc_folds_kfold.json \
    --out /ix1/pcody/cellpose/data/CLC_full2/clc_folds_kfold_v2.json'
```

## 3. Materialize per-fold train/test dirs for a recipe

```bash
MAN=/ix1/pcody/cellpose/data/CLC_full/clc_folds_kfold.json
ssh pitt_crc "module use ~/modulefiles; module load cellpose_env; cd ~/cellpose; \
  for r in clc_all clc_adult clc_neonate; do \
    python clc_cv.py materialize --manifest $MAN --recipe \$r --out_root runs/clc_cv_full; \
  done"
# recipes: clc_all | clc_adult | clc_neonate | cunningham_plus_clc(--base_train DIR)
```

## 4. Train a recipe (5-fold CV, one SLURM array)

`submit_cv.sh <recipe> [INIT]` reads the materialized fold dirs and submits an
array (one task/fold). Env overrides:
- `FOLD_ROOT` — where the `fold_<k>/` dirs are (default `runs/clc_cv/<recipe>`).
- `RUN_TAG` — names the models `<RUN_TAG>_fold<k>` + the CSV dir.
- `CELLPOSE_TRAIN_CONFIG` — which yaml (default `trainer.yaml` = from-cpsam).
- `INIT` (2nd arg) — warm-start model path (else from stock cpsam).

```bash
ssh pitt_crc 'cd ~/cellpose; \
  FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full/clc_all  RUN_TAG=clc_all_full_20260604 \
    ./submit_cv.sh clc_all'                                  # mixed, from cpsam

# age-specific (same config => only age composition differs):
ssh pitt_crc 'cd ~/cellpose; \
  FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full/clc_adult RUN_TAG=clc_adult_full_20260604 ./submit_cv.sh clc_adult'
ssh pitt_crc 'cd ~/cellpose; \
  FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full/clc_neonate RUN_TAG=clc_neonate_full_20260604 ./submit_cv.sh clc_neonate'

# warm-start (refine the Cunningham model on CLC):
ssh pitt_crc 'cd ~/cellpose; \
  CELLPOSE_TRAIN_CONFIG=$HOME/cellpose/trainer_warmstart.yaml \
  FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full/clc_all RUN_TAG=clc_all_warm_full \
    ./submit_cv.sh clc_all /ix1/pcody/cellpose/models/label_xfer_aug_retest'

# FN-focused augmentation (alt config, same data):
ssh pitt_crc 'cd ~/cellpose; \
  CELLPOSE_TRAIN_CONFIG=$HOME/cellpose/trainer_fnaug.yaml \
  FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full/clc_all RUN_TAG=clc_all_fnaug_20260604 \
    ./submit_cv.sh clc_all'
```

Single (non-CV) train from a config: edit `trainer.yaml` (`data.source`,
`train.*`, `augment:`) then `cd ~/cellpose && ./submit.sh` (uses `run_trainer.slurm`).

### 4a. Production deploy model — `clc_all_full_deploy` (the model in prod)

Train the winning recipe on **ALL 67 images, no held-out fold**. Stage a deploy
dir whose `train/` **and** `test/` are symlinks to every image (a deploy model
holds nothing out; `test/` exists only because `run_trainer.slurm` requires it,
and gives in-sample loss monitoring), then submit with `trainer_deploy.yaml`.

**Step 1 — push the config + (re)build the deploy dir, and verify:**
```bash
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
```
Expected: `train: seg=67 tif=67` / `test: seg=67 tif=67` and
`/ix1/pcody/cellpose/data/CLC_full_deploy | model: clc_all_full_deploy`.
(`rm -rf $DEP` only removes the symlink farm — `CLC_full/` holds the real data.
Rebuilding it is how new labeled images get picked up; re-verify the counts.)

**Step 2 — submit the training job:**
```bash
ssh pitt_crc 'cd ~/cellpose && sbatch \
  --export=ALL,CELLPOSE_TRAIN_CONFIG=$HOME/cellpose/trainer_deploy.yaml run_trainer.slurm'
# 1 h 37 m on an A100 (job 2702929) -> /ix1/pcody/cellpose/models/clc_all_full_deploy
# frozen config: ~/cellpose/run_logs/cellpose_train_<jobid>.config.yaml
```

**Step 3 — register locally for inference / the GUI:**
```bash
python -m cellpose --add_model /path/to/clc_all_full_deploy
```

Every trained artifact and the exact code that produced it is recorded in
[`model_train_log.md`](model_train_log.md) — **add an entry there for any new
model.**

### 4c. "Does a new data batch help?" — the paired two-arm test

Never judge new data by comparing a fresh CV mean against the old one — the test
sets differ, so you can't separate "more data helped" from "the test set changed".
Instead, with the fold assignment pinned by `--extend` (§2), run:

```bash
# arm B — retrain the same recipe on the bigger set
ssh pitt_crc 'cd ~/cellpose; \
  python clc_cv.py materialize --manifest $DATA/CLC_full2/clc_folds_kfold_v2.json \
    --recipe clc_all --out_root runs/clc_cv_full2; \
  FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full2/clc_all RUN_TAG=clc_all_full2_<date> \
    ./submit_cv.sh clc_all'

# arm A — the EXISTING fold models, re-scored on the extended test folds
#          (inference only, no retraining): run_control_eval.slurm
ssh pitt_crc 'cd ~/cellpose && sbatch run_control_eval.slurm'

# watcher: waits for both, pools the paired comparison, submits the error-eval
ssh pitt_crc 'cd ~/cellpose && screen -dmS full2 ./finish_full2.sh'
```
Arm A is leak-free on the new images (it never trained on any of them) and on the
old ones (unchanged folds). Both arms use `nimg_per_epoch: 750`, so the gradient-
step budget is identical and the only variable is the sampling pool. Always check
the harness first: arm A's re-scored **old** images must reproduce the previous
run's CSVs exactly (§4a of `model_train_log.md` — it did, max|Δ| = 0).

### 4b. Alternate-backbone arms (cpdino / cpsam_v2) — separate conda env

The backbone comparison (2026-06-23) runs upstream-main cellpose in its own
module env (`cellpose_dino_env`, built by `setup_dino_env.sh` / `setup_v2_env.sh`)
with a **stock** trainer (`trainer_slurm_stock.py` — no online-aug hook, no
boundary loss), driven by `run_clc_cv_dino.slurm` instead of `run_clc_cv.slurm`.
Same fold dirs, same `eval_seg.py`. Backbone is chosen by `INIT`:

```bash
ssh pitt_crc 'cd ~/cellpose; \
  RECIPE=clc_all FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full/clc_all \
  RUN_TAG=clc_all_dino_stock_20260623 INIT=$HOME/.cellpose/models/cpdino \
  sbatch --array=0-4 --export=ALL,RECIPE,FOLD_ROOT,RUN_TAG,INIT run_clc_cv_dino.slurm'
# INIT=$HOME/.cellpose/models/cpsam_v2      -> cpsam_v2 arm
# CELLPOSE_TRAIN_CONFIG=trainer_dino384.yaml -> cpdino at its native 384 crop (bsize 384, batch 4)
# then: ./finish_dino.sh   (waits, pools paired CV AP@0.5, submits run_erreval_dino.slurm)
```

> These scripts live **only on the cluster** (`~/cellpose/`), not in the repo —
> see [`model_train_log.md`](model_train_log.md) §4. All arms tied with the
> cpsam baseline; nothing here needs re-running.

## 5. Evaluate

```bash
MAN=/ix1/pcody/cellpose/data/CLC_full/clc_folds_kfold.json
ADULT=/ix1/pcody/cellpose/data/CLC_full/adult
NEO=/ix1/pcody/cellpose/data/CLC_full/neonate

# 5a. AP@IoU pooled over held-out folds (per-image CSVs already written by step 4)
ssh pitt_crc "module use ~/modulefiles; module load cellpose_env; cd ~/cellpose; \
  python clc_cv.py aggregate --manifest $MAN \
    --results_dir run_logs/clc_all_full_20260604 \
    [--baseline_csv run_logs/eval_<job>.csv]"        # baseline => paired Wilcoxon

# 5b. Baseline AP of any existing models on a test set (no training)
ssh pitt_crc "cd ~/cellpose; TEST_DIR='$ADULT,$NEO' \
  MODELS='cpsam,label_xfer_aug_retest' sbatch --export=ALL,TEST_DIR,MODELS run_eval.slurm"

# 5c. Curation-as-error-set: FN-recovery / FP-suppression / TP-retention (Approach B)
ssh pitt_crc "cd ~/cellpose; \
  MODELS='cpsam,heldout:$MODELS/clc_all_full_20260604_fold{k}' \
  DATA_DIRS='$ADULT,$NEO' MANIFEST=$MAN \
  sbatch --export=ALL,MODELS,DATA_DIRS,MANIFEST run_erreval.slurm"
# age-restricted three-way (adult): include both specialist + mixed models, adult dir only
ssh pitt_crc "cd ~/cellpose; \
  MODELS='cpsam,heldout:/ix1/pcody/cellpose/models/clc_adult_full_20260604_fold{k},heldout:/ix1/pcody/cellpose/models/clc_all_full_20260604_fold{k}' \
  DATA_DIRS='$ADULT' MANIFEST=$MAN \
  sbatch --export=ALL,MODELS,DATA_DIRS,MANIFEST run_erreval.slurm"

# 5d. Over-detect threshold sweep (FN-recovery vs FP vs AP over a flow x cellprob grid)
ssh pitt_crc "cd ~/cellpose; \
  MODEL_PATTERN='/ix1/pcody/cellpose/models/clc_all_full_20260604_fold{k}' \
  sbatch --export=ALL,MODEL_PATTERN run_sweep.slurm"
```

### Paired per-image comparison (e.g. specialist vs mixed on one age)
`clc_cv.py aggregate` collapses `_fold<k>` and does the paired test vs a baseline
CSV. For a two-run comparison on matching images, pair per-image AP from the two
runs' `eval_fold*.csv` (see the snippet pattern in `clc_seg_results.md` §results
and the Result-5/6 comparisons in `progress.md`).

---

## 6. Monitor / collect

```bash
ssh pitt_crc 'squeue -M gpu -u $USER -o "%i %T %M %N"'        # queue
ssh pitt_crc 'tail -f ~/cellpose/run_logs/clc_cv_<JOBID>_0.out'
ls ~/cellpose/run_logs/<RUN_TAG>/eval_fold*.csv               # per-image results
ls /ix1/pcody/cellpose/models/<RUN_TAG>_fold*                 # trained models
```

Notes: SLURM block-buffers stdout (`.out` lags ~minutes). Each array task copies
its model back to `$MODELS` on exit; a failed (e.g. OOM) fold leaves no model and
is failed loudly by the guard in `run_clc_cv.slurm`.

---

## Worked examples actually run (job IDs for traceability)

| experiment | command (recipe / config) | job |
|---|---|---|
| **PROD deploy model** | `run_trainer.slurm` (trainer_deploy.yaml) — §4a | **2702929** |
| full-set mixed CV (the control) | `submit_cv.sh clc_all` (trainer.yaml) | 2700928 |
| full-set FN-recovery eval | `run_erreval.slurm` MODELS=clc_all_full | 2701730 |
| FN-aug | `submit_cv.sh clc_all` (trainer_fnaug.yaml) | 2701806 |
| age: adult-only | `submit_cv.sh clc_adult` | 2701818 |
| age: neonate-only | `submit_cv.sh clc_neonate` | 2701823 |
| boundary loss (α=5) | `submit_cv.sh clc_all` (trainer_bndry.yaml) | 2702487 |
| backbone: cpsam_v2 + fork aug | `run_clc_cv.slurm` INIT=cpsam_v2 | 2702880 |
| backbone: cpdino (stock env) | `run_clc_cv_dino.slurm` INIT=cpdino | 2702957 |
| backbone: cpsam_v2 (stock env) | `run_clc_cv_dino.slurm` INIT=cpsam_v2 | 2702958 |
| backbone: cpdino @384 crop | `run_clc_cv_dino.slurm` (trainer_dino384.yaml) | 2703588 |
| backbone FN-recovery eval | `run_erreval_dino.slurm` | 2703245 |
| +20 imgs (2026-07) arm B | `submit_cv.sh clc_all` on CLC_full2 | 3346056 |
| +20 imgs arm A (control re-score) | `run_control_eval.slurm` | 3346061 |
| +20 imgs FN-recovery, both arms | `run_erreval.slurm` (87 imgs) | 3346128 |
| 15-img LOAO mixed | `submit_cv.sh clc_all` (clc_folds_loao) | 2646810 |
| warm-start | `submit_cv.sh clc_all <init>` (trainer_warmstart) | 2647218 |
| threshold sweep | `run_sweep.slurm` | 2699771 |

See `model_train_log.md` (per-model training code + artifacts), `progress.md`
(status), `clc_seg_results.md` (results), `clc_seg_experiments.md` (method/plan).
