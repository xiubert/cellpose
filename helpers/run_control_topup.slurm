#!/bin/bash
#SBATCH --job-name=clc_ctrl_top
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=run_logs/clc_ctrl_top_%j.out
#SBATCH --error=run_logs/clc_ctrl_top_%j.err
#
# Control top-ups for arms E and F. The control (clc_all_full_20260604_fold<k>)
# was already scored on all 98 CLC_full3 images (run_logs/..._v4eval). Two gaps:
#
#  E) the 20x image is new to the comparison set  -> score it with the fold-3
#     control model (20x belongs to animal 8363, fold 3).
#  F) 8483 moved from fold 3 to fold 0 in the v6 split -> its 3 images must be
#     scored by the fold-0 control model so the pairing matches the new mapping.
#     (Any control fold model is leak-free on 8483 — the control never trained on
#     ANY new image — but the fold used must match arm F's, or the two arms are
#     not compared under the same held-out condition.)

module use ~/modulefiles
module load cellpose_env
module load cuda/12.4.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/cellpose"
D=/ix1/pcody/cellpose/data
MODELS_DIR=/ix1/pcody/cellpose/models

# E: 20x only
mkdir -p "$SLURM_SCRATCH/x20"
cp -L "$D"/CLC_full4/adult/*20x*_seg.npy "$D"/CLC_full4/adult/*20x*.tif "$SLURM_SCRATCH/x20/"
echo "=== control fold3 on the 20x image ==="
python eval_seg.py --test_dir "$SLURM_SCRATCH/x20" \
    --models clc_all_full_20260604_fold3 --models_dir "$MODELS_DIR" \
    --out run_logs/ctrl_fold3_on_20x.csv

# F: 8483 under its new fold-0 assignment
mkdir -p "$SLURM_SCRATCH/a8483"
cp -L "$D"/CLC_full3/adult/*8483*_seg.npy "$D"/CLC_full3/adult/*8483*.tif "$SLURM_SCRATCH/a8483/"
echo "=== control fold0 on 8483 (3 images) ==="
python eval_seg.py --test_dir "$SLURM_SCRATCH/a8483" \
    --models clc_all_full_20260604_fold0 --models_dir "$MODELS_DIR" \
    --out run_logs/ctrl_fold0_on_8483.csv

# also: the DEPLOYED model on the 20x image (never trained on it -> leak-free)
echo "=== deployed model on the 20x image ==="
python eval_seg.py --test_dir "$SLURM_SCRATCH/x20" \
    --models clc_all_full_deploy --models_dir "$MODELS_DIR" \
    --out run_logs/deploy_on_20x.csv
echo "Done."
