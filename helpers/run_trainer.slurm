#!/bin/bash
#SBATCH --job-name=cellpose_train
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --qos=long
#SBATCH --output=run_logs/cellpose_train_%j.out
#SBATCH --error=run_logs/cellpose_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# --mail-user is supplied at submit time by submit.sh (from .env / PITT_EMAIL);
# submit with:  ./submit.sh   (not: sbatch run_trainer.slurm)

# for L40S, we use fewer resources (4 CPUs, 32 GB RAM) than for A100, since the GPU is more powerful and can handle the training load without needing as much CPU/RAM support. Adjust these values if you find that the job is running out of memory or if you want to speed up data loading.
# --partition=l40s
# --gres=gpu:1
# --constraint=l40s,48g,intel
# --cpus-per-task=4
# --mem=32G

# --- Environment ---
module use ~/modulefiles
module load cellpose_env
module load cuda/12.4.1

# --- Data: copy from /ix1 project storage to fast local scratch ---
# Data should live in /ix1/<group>/cellpose/ — NOT in $HOME (75 GB quota, not for compute data)
DATA_SRC="/ix1/$(id -gn)/cellpose"  # resolves to /ix1/<groupId>/cellpose
mkdir -p "$SLURM_SCRATCH/train" "$SLURM_SCRATCH/test"
if [ -z "$(ls -A "$SLURM_SCRATCH/train")" ]; then
    echo "Copying train data to scratch (${DATA_SRC}/train/)..."
    cp -r "$DATA_SRC/train/." "$SLURM_SCRATCH/train/"
else
    echo "Train data already in scratch, skipping copy."
fi
if [ -z "$(ls -A "$SLURM_SCRATCH/test")" ]; then
    echo "Copying test data to scratch (${DATA_SRC}/test/)..."
    cp -r "$DATA_SRC/test/." "$SLURM_SCRATCH/test/"
else
    echo "Test data already in scratch, skipping copy."q
fi

# --- Point trainer at scratch directories ---
export CELLPOSE_TRAIN_DIR="$SLURM_SCRATCH/train"
export CELLPOSE_TEST_DIR="$SLURM_SCRATCH/test"

# --- Hyperparameter config (single source of truth; see trainer.yaml) ---
export CELLPOSE_TRAIN_CONFIG="${CELLPOSE_TRAIN_CONFIG:-$HOME/cellpose/trainer.yaml}"

# --- Dump train/test input filenames + freeze the config to job-tagged logs ---
INPUT_LOG="run_logs/cellpose_train_${SLURM_JOB_ID}.inputs"
mkdir -p run_logs
cp "$CELLPOSE_TRAIN_CONFIG" "run_logs/cellpose_train_${SLURM_JOB_ID}.config.yaml" 2>/dev/null \
    || echo "warning: config $CELLPOSE_TRAIN_CONFIG not found; trainer will use built-in defaults"
{
    echo "=== Train inputs (${CELLPOSE_TRAIN_DIR} via ${DATA_SRC}/train/) ==="
    ls "$CELLPOSE_TRAIN_DIR"
    echo ""
    echo "=== Test inputs (${CELLPOSE_TEST_DIR} via ${DATA_SRC}/test/) ==="
    ls "$CELLPOSE_TEST_DIR"
} > "$INPUT_LOG"

# --- Trap: copy model output back to project storage on exit (success or failure) ---
run_on_exit() {
    echo "Copying models back to project storage..."
    mkdir -p "$DATA_SRC/models"
    cp -r "$HOME/cellpose/models/." "$DATA_SRC/models/" 2>/dev/null || true
}
trap run_on_exit EXIT

# --- Run ---
echo "Starting training on node: $(hostname)"
echo "GPU info:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python "$HOME/cellpose/trainer_slurm.py"

# Parse training log and write CSV + PNG alongside the .out file
LOG_FILE="run_logs/cellpose_train_${SLURM_JOB_ID}.out"
python "$HOME/cellpose/helpers/parse_training_log.py" "$LOG_FILE"

crc-job-stats.py

echo "Done."
