#!/bin/bash
#SBATCH --job-name=clc_ctrl3
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=run_logs/clc_ctrl3_%j.out
#SBATCH --error=run_logs/clc_ctrl3_%j.err
#
# CONTROL ARM (A') for the 2026-08 batch experiment, on CLC_full3 (98 images).
#
# Two evaluations:
#  1. clc_all_full_20260604_fold<k> on fold k's held-out images — the leak-free
#     CV twin of the deployed model, the paired baseline for arm D. This MUST be
#     re-run rather than reused: 6 images were re-curated since the last round
#     (4 of the original 67 + 4L 16khz + BL 16khz), so their previous AP values
#     are scored against stale ground truth.
#  2. clc_all_full_deploy itself on the 31 images from adult(2)/neonate(2) —
#     images the SHIPPED model has never trained on, so this read is leak-free
#     and answers "what does production actually do on the new data today".

module use ~/modulefiles
module load cellpose_env
module load cuda/12.4.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/cellpose"
D=/ix1/pcody/cellpose/data
FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full3/clc_all
OUT_DIR=run_logs/clc_all_full_20260604_v4eval
MODELS_DIR=/ix1/pcody/cellpose/models
mkdir -p "$OUT_DIR"

for k in 0 1 2 3 4; do
    echo "=== control fold $k on $(ls $FOLD_ROOT/fold_$k/test/*_seg.npy | wc -l) held-out images ==="
    python eval_seg.py \
        --test_dir "$FOLD_ROOT/fold_$k/test" \
        --models "clc_all_full_20260604_fold$k" \
        --models_dir "$MODELS_DIR" \
        --out "$OUT_DIR/eval_fold$k.csv"
done

echo "=== deployed model on the 31 never-seen new images ==="
python eval_seg.py \
    --test_dir "$D/CLC_full3_newonly" \
    --models "clc_all_full_deploy" \
    --models_dir "$MODELS_DIR" \
    --out "run_logs/deploy_on_newonly.csv"

echo "Done."
