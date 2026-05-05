#!/bin/bash
#SBATCH --job-name=cellpose_train
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s,48g,intel
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --qos=long
#SBATCH --output=cellpose_train_%j.out
#SBATCH --error=cellpose_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=[your_pitt_email]@pitt.edu

# --- Environment ---
module use ~/modulefiles
module load cellpose_env
module load cuda/12.4.1

# --- Data: copy from /ix1 project storage to fast local scratch ---
# Data should live in /ix1/<group>/cellpose/ — NOT in $HOME (75 GB quota, not for compute data)
DATA_SRC="/ix1/$(id -gn)/cellpose"  # resolves to /ix1/<groupId>/cellpose
mkdir -p "$SLURM_SCRATCH/train" "$SLURM_SCRATCH/test"
if [ -z "$(ls -A "$SLURM_SCRATCH/train")" ]; then
    echo "Copying train data to scratch..."
    cp -r "$DATA_SRC/train/." "$SLURM_SCRATCH/train/"
else
    echo "Train data already in scratch, skipping copy."
fi
if [ -z "$(ls -A "$SLURM_SCRATCH/test")" ]; then
    echo "Copying test data to scratch..."
    cp -r "$DATA_SRC/test/." "$SLURM_SCRATCH/test/"
else
    echo "Test data already in scratch, skipping copy."
fi

# --- Point trainer at scratch directories ---
export CELLPOSE_TRAIN_DIR="$SLURM_SCRATCH/train"
export CELLPOSE_TEST_DIR="$SLURM_SCRATCH/test"

# --- Trap: copy model output back to project storage on exit (success or failure) ---
run_on_exit() {
    echo "Copying models back to project storage..."
    mkdir -p "$DATA_SRC/models"
    cp -r "$SLURM_SCRATCH/train/models/." "$DATA_SRC/models/" 2>/dev/null || true
}
trap run_on_exit EXIT

# --- Run ---
echo "Starting training on node: $(hostname)"
echo "GPU info:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python "$HOME/cellpose/trainer_slurm.py"

crc-job-stats.py

echo "Done."
