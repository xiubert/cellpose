#!/bin/bash
#SBATCH --job-name=clc_ctrl_eval
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=run_logs/clc_ctrl_eval_%j.out
#SBATCH --error=run_logs/clc_ctrl_eval_%j.err
#
# CONTROL ARM (arm A) of the "does the 2026-07 data help?" experiment.
#
# Scores the DEPLOYED-recipe CV models (clc_all_full_20260604_fold<k>, trained on
# the old 67 images only) on the EXTENDED fold-k test sets (old + new images).
# Leak-free: fold k's control model never trained on fold k's animals, and it
# never saw ANY of the new images. Fold assignment for old animals is identical
# in both manifests (clc_split.py --extend), so this is a paired comparison
# against clc_all_full2_20260731_fold<k> on exactly the same images.
#
# Re-scoring the old images too is a deliberate harness check: those AP values
# must reproduce run_logs/clc_all_full_20260604/eval_fold*.csv.

module use ~/modulefiles
module load cellpose_env
module load cuda/12.4.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/cellpose"
FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full2/clc_all
OUT_DIR=run_logs/clc_all_full_20260604_v2eval
MODELS_DIR=/ix1/pcody/cellpose/models
mkdir -p "$OUT_DIR"

for k in 0 1 2 3 4; do
    echo "=== control fold $k: clc_all_full_20260604_fold$k on $(ls $FOLD_ROOT/fold_$k/test/*_seg.npy | wc -l) held-out images ==="
    python eval_seg.py \
        --test_dir "$FOLD_ROOT/fold_$k/test" \
        --models "clc_all_full_20260604_fold$k" \
        --models_dir "$MODELS_DIR" \
        --out "$OUT_DIR/eval_fold$k.csv"
done
echo "Done."
