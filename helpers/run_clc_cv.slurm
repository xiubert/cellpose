#!/bin/bash
#SBATCH --job-name=clc_cv
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --qos=long
#SBATCH --output=run_logs/clc_cv_%A_%a.out
#SBATCH --error=run_logs/clc_cv_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# Leave-one-animal-out CV for CLC segmentation — one ARRAY TASK per fold.
# Submit with submit_cv.sh (sets --array from the materialized fold dirs).
#
# Per task ($SLURM_ARRAY_TASK_ID = fold index):
#   1. stage runs/clc_cv/<RECIPE>/fold_<k>/{train,test} to node-local scratch
#   2. train  -> model <RECIPE>_fold<k>  (warm-start if INIT set)
#   3. eval   -> run_logs/<RUN_TAG>/eval_fold<k>.csv  (held-out animal)
#
# Env (exported by submit_cv.sh / --export):
#   RECIPE      data recipe (clc_all | clc_adult | clc_neonate | cunningham_plus_clc)
#   FOLD_ROOT   dir holding fold_<k>/ (default: $HOME/cellpose/runs/clc_cv/$RECIPE)
#   RUN_TAG     subdir under run_logs for this CV run's per-fold CSVs
#   INIT        (optional) warm-start model path or name (e.g. label_xfer_aug_retest)
#   MODELS_DIR  where trained models live / are read for eval (default /ix1/.../models)
#   EVAL_FIXED_SCALE  if set, eval with --fixed_scale <v>; MUST match the trainer
#                     yaml's train.fixed_scale or the model is scored under a
#                     different input distribution than it was trained on.

module use ~/modulefiles
module load cellpose_env
module load cuda/12.4.1

# Reduce CUDA fragmentation OOMs (seen when an array task shares a busy GPU).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/cellpose"
mkdir -p run_logs

FOLD="${SLURM_ARRAY_TASK_ID:?array task id missing — submit with --array}"
RECIPE="${RECIPE:?set RECIPE}"
FOLD_ROOT="${FOLD_ROOT:-$HOME/cellpose/runs/clc_cv/$RECIPE}"
RUN_TAG="${RUN_TAG:-${RECIPE}_$(date +%Y%m%d)}"
MODELS_DIR="${MODELS_DIR:-/ix1/$(id -gn)/cellpose/models}"
SRC="$FOLD_ROOT/fold_${FOLD}"
OUT_CSV_DIR="run_logs/$RUN_TAG"
mkdir -p "$OUT_CSV_DIR"

echo "fold=$FOLD recipe=$RECIPE src=$SRC run_tag=$RUN_TAG init=${INIT:-<cpsam>}"
[ -d "$SRC/train" ] || { echo "ERROR: $SRC/train missing — run clc_cv.py materialize first"; exit 1; }

# --- stage to scratch (cp -L resolves the symlinks the materialize step made) ---
mkdir -p "$SLURM_SCRATCH/train" "$SLURM_SCRATCH/test"
cp -rL "$SRC/train/." "$SLURM_SCRATCH/train/"
cp -rL "$SRC/test/."  "$SLURM_SCRATCH/test/"
echo "train files: $(ls "$SLURM_SCRATCH/train"/*_seg.npy | wc -l)  test files: $(ls "$SLURM_SCRATCH/test"/*_seg.npy | wc -l)"

export CELLPOSE_TRAIN_DIR="$SLURM_SCRATCH/train"
export CELLPOSE_TEST_DIR="$SLURM_SCRATCH/test"
# Model name is tagged by RUN_TAG (not just recipe) so different CV runs of the
# same recipe — e.g. from-cpsam vs warm-start — never collide / overwrite each
# other, and so a failed fold's eval can't silently pick up a stale same-named
# model from a previous run.
export CELLPOSE_TRAIN_CONFIG="${CELLPOSE_TRAIN_CONFIG:-$HOME/cellpose/trainer.yaml}"
export CELLPOSE_MODEL_NAME="${RUN_TAG}_fold${FOLD}"
[ -n "$INIT" ] && export CELLPOSE_PRETRAINED="$INIT"

# freeze the resolved config for this fold
cp "$CELLPOSE_TRAIN_CONFIG" "$OUT_CSV_DIR/config_fold${FOLD}.yaml" 2>/dev/null || true

# trainer_slurm saves the model to ~/cellpose/models (cwd/models). HOME has a
# 75 GB quota — accumulating 1.2 GB fold models there across runs fills it and
# torch.save fails ("File ... cannot be opened"). So we move each model to /ix1
# (5 TB) and free HOME immediately after training. Trap = safety-net cleanup.
trap 'rm -f "$HOME/cellpose/models/$CELLPOSE_MODEL_NAME" 2>/dev/null || true' EXIT

# --- train ---
echo "=== training $CELLPOSE_MODEL_NAME ==="
python "$HOME/cellpose/trainer_slurm.py"

# Guard: if training failed (e.g. CUDA OOM / quota), the model file won't exist —
# fail the task loudly rather than evaluating a stale/wrong model.
if [ ! -f "$HOME/cellpose/models/$CELLPOSE_MODEL_NAME" ]; then
    echo "ERROR: model $HOME/cellpose/models/$CELLPOSE_MODEL_NAME not produced — training failed; skipping eval"
    exit 1
fi

# Move model off HOME -> /ix1 right away (durable, frees the home quota).
mkdir -p "$MODELS_DIR"
cp "$HOME/cellpose/models/$CELLPOSE_MODEL_NAME" "$MODELS_DIR/" \
    && rm -f "$HOME/cellpose/models/$CELLPOSE_MODEL_NAME"

# --- eval the just-trained fold model on the held-out test set ---
echo "=== evaluating $CELLPOSE_MODEL_NAME on held-out animal ==="
FS_ARG=""
[ -n "${EVAL_FIXED_SCALE:-}" ] && FS_ARG="--fixed_scale $EVAL_FIXED_SCALE"
python "$HOME/cellpose/eval_seg.py" $FS_ARG \
    --test_dir "$CELLPOSE_TEST_DIR" \
    --models "$CELLPOSE_MODEL_NAME" \
    --models_dir "$MODELS_DIR" \
    --out "$OUT_CSV_DIR/eval_fold${FOLD}.csv"

echo "Done fold $FOLD."
