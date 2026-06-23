#!/bin/bash
#SBATCH --job-name=clc_sweep
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=run_logs/clc_sweep_%j.out
#SBATCH --error=run_logs/clc_sweep_%j.err
#
# Over-detect threshold sweep (clc_threshold_sweep.py): FN-recovery / TP-ret /
# model-FP / AP@0.5 over a flow_threshold x cellprob_threshold grid on the LOAO
# /kfold held-out fold models. Override via env at submit:
#   MODEL_PATTERN (required) held-out model path with {k}
#   DATA_DIRS / MANIFEST   (defaults: full set)
#   CELLPROB / FLOW        comma grids (defaults below)
# Example:
#   MODEL_PATTERN='/ix1/pcody/cellpose/models/clc_all_full_20260604_fold{k}' \
#     sbatch --export=ALL,MODEL_PATTERN run_sweep.slurm

module use ~/modulefiles; module load cellpose_env; module load cuda/12.4.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/cellpose"; mkdir -p run_logs

DATA_DIRS="${DATA_DIRS:-/ix1/pcody/cellpose/data/CLC_full/adult,/ix1/pcody/cellpose/data/CLC_full/neonate}"
MANIFEST="${MANIFEST:-/ix1/pcody/cellpose/data/CLC_full/clc_folds_kfold.json}"
CELLPROB="${CELLPROB:-0.0,-1.0,-2.0,-4.0,-6.0}"
FLOW="${FLOW:-0.4,0.8}"
: "${MODEL_PATTERN:?set MODEL_PATTERN (e.g. '/ix1/.../models/<tag>_fold{k}')}"

python "$HOME/cellpose/clc_threshold_sweep.py" \
    --data_dirs "$DATA_DIRS" --manifest "$MANIFEST" \
    --model_pattern "$MODEL_PATTERN" --cellprob "$CELLPROB" --flow "$FLOW"
echo "Done."
