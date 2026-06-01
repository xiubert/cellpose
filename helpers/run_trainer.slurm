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

# --- Hyperparameter + dataset config (single source of truth; see trainer.yaml) ---
export CELLPOSE_TRAIN_CONFIG="${CELLPOSE_TRAIN_CONFIG:-$HOME/cellpose/trainer.yaml}"

# Read a key from the trainer.yaml `data:` section (PyYAML ships with cellpose_env).
yaml_data_get() {
    python -c "import yaml; d=(yaml.safe_load(open('$CELLPOSE_TRAIN_CONFIG')) or {}).get('data') or {}; print(d.get('$1') or '')" 2>/dev/null
}

# --- Data: copy from /ix1 project storage to fast local scratch ---
# The dataset dir (must contain train/ and test/ with paired .tif + _seg.npy)
# comes from trainer.yaml `data.source`, so the dataset is part of the frozen,
# reproducible run config — not hardcoded here. CELLPOSE_DATA_SRC overrides;
# final fallback is the PMID-38653806 / Cunningham label-transfer set.
DATA_SRC="${CELLPOSE_DATA_SRC:-$(yaml_data_get source)}"
DATA_SRC="${DATA_SRC:-/ix1/$(id -gn)/cellpose/data/PMID_38653806/Cunningham}"
# Trained models are copied back here, kept out of the data dir. From
# trainer.yaml `data.model_dst`, else CELLPOSE_MODEL_DST, else the canonical dir.
MODEL_DST="${CELLPOSE_MODEL_DST:-$(yaml_data_get model_dst)}"
MODEL_DST="${MODEL_DST:-/ix1/$(id -gn)/cellpose/models}"
echo "DATA_SRC=$DATA_SRC"
echo "MODEL_DST=$MODEL_DST"
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
    echo "Test data already in scratch, skipping copy."
fi

# --- Point trainer at scratch directories ---
export CELLPOSE_TRAIN_DIR="$SLURM_SCRATCH/train"
export CELLPOSE_TEST_DIR="$SLURM_SCRATCH/test"

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
    echo "Copying models back to project storage ($MODEL_DST)..."
    mkdir -p "$MODEL_DST"
    cp -r "$HOME/cellpose/models/." "$MODEL_DST/" 2>/dev/null || true
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
