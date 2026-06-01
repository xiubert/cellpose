#!/bin/bash
#SBATCH --job-name=cellpose_eval
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=run_logs/cellpose_eval_%j.out
#SBATCH --error=run_logs/cellpose_eval_%j.err
#
# Head-to-head segmentation eval (AP@IoU) on a test dir, for one or more models.
# Reads test data read-only from /ix1 (no scratch staging needed for ~23 imgs).
# Override MODELS / TEST_DIR / MODELS_DIR via env at submit time if desired.

module use ~/modulefiles
module load cellpose_env
module load cuda/12.4.1

cd "$HOME/cellpose"
mkdir -p run_logs

TEST_DIR="${TEST_DIR:-/ix1/pcody/cellpose/data/PMID_38653806/Cunningham/test}"
MODELS="${MODELS:-label_xfer_aug_retest,label_xfer_online_aug}"
MODELS_DIR="${MODELS_DIR:-/ix1/pcody/cellpose/models}"

echo "TEST_DIR=$TEST_DIR"
echo "MODELS=$MODELS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "$HOME/cellpose/eval_seg.py" \
    --test_dir "$TEST_DIR" \
    --models "$MODELS" \
    --models_dir "$MODELS_DIR" \
    --out "run_logs/eval_${SLURM_JOB_ID}.csv"

echo "Done."
