#!/bin/bash
#SBATCH --job-name=clc_err
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=run_logs/clc_err_%j.out
#SBATCH --error=run_logs/clc_err_%j.err
#
# Curation-as-error-set eval (Approach B): FN-recovery / FP-suppression /
# TP-retention vs cpsam, via clc_error_eval.py. Override via env at submit:
#   MODELS     (required) comma list; "cpsam", a path, or
#              "heldout:<pattern-with-{k}>" for the LOAO/kfold held-out model.
#   DATA_DIRS  comma-separated CLC dirs (default: full set adult,neonate)
#   MANIFEST   fold manifest json (default: CLC_full kfold)
# Example:
#   MODELS='cpsam,heldout:/ix1/pcody/cellpose/models/clc_all_full_20260604_fold{k}' \
#     sbatch --export=ALL,MODELS run_erreval.slurm

module use ~/modulefiles; module load cellpose_env; module load cuda/12.4.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/cellpose"; mkdir -p run_logs

DATA_DIRS="${DATA_DIRS:-/ix1/pcody/cellpose/data/CLC_full/adult,/ix1/pcody/cellpose/data/CLC_full/neonate}"
MANIFEST="${MANIFEST:-/ix1/pcody/cellpose/data/CLC_full/clc_folds_kfold.json}"
: "${MODELS:?set MODELS (e.g. 'cpsam,heldout:/ix1/.../models/<tag>_fold{k}')}"

echo "DATA_DIRS=$DATA_DIRS"; echo "MANIFEST=$MANIFEST"; echo "MODELS=$MODELS"
python "$HOME/cellpose/clc_error_eval.py" \
    --data_dirs "$DATA_DIRS" --manifest "$MANIFEST" --models "$MODELS"
echo "Done."
