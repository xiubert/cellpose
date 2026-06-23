#!/bin/bash
# Submit a LOAO CV run (run_clc_cv.slurm) as a SLURM array — one task per
# materialized fold. Derives the --array list from the fold_<k> dirs that
# clc_cv.py materialize created (so age-restricted recipes submit only their
# active folds). Reads PITT_EMAIL from .env like submit.sh.
#
# Usage:
#   ./submit_cv.sh <recipe> [INIT]
#     recipe : clc_all | clc_adult | clc_neonate | cunningham_plus_clc
#     INIT   : optional warm-start model path/name (e.g. label_xfer_aug_retest)
#
# Example:
#   ./submit_cv.sh clc_all
#   ./submit_cv.sh clc_all label_xfer_aug_retest      # warm-start refine
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="${1:?usage: $0 <recipe> [INIT]}"
INIT="${2:-}"
FOLD_ROOT="${FOLD_ROOT:-$here/runs/clc_cv/$RECIPE}"

[ -d "$FOLD_ROOT" ] || { echo "ERROR: $FOLD_ROOT not found — run clc_cv.py materialize --recipe $RECIPE first" >&2; exit 1; }

# array list = the fold indices that actually exist
folds=$(ls -d "$FOLD_ROOT"/fold_* 2>/dev/null | sed 's/.*fold_//' | sort -n | paste -sd, -)
[ -n "$folds" ] || { echo "ERROR: no fold_* dirs in $FOLD_ROOT" >&2; exit 1; }

RUN_TAG="${RUN_TAG:-${RECIPE}$( [ -n "$INIT" ] && echo _warm )_$(date +%Y%m%d_%H%M)}"

[ -f "$here/.env" ] && { set -a; . "$here/.env"; set +a; }
: "${PITT_EMAIL:?set PITT_EMAIL in .env}"

echo "recipe=$RECIPE  folds=[$folds]  run_tag=$RUN_TAG  init=${INIT:-<cpsam>}"
sbatch --array="$folds" --mail-user="$PITT_EMAIL" \
    --export=ALL,RECIPE="$RECIPE",FOLD_ROOT="$FOLD_ROOT",RUN_TAG="$RUN_TAG",INIT="$INIT" \
    "$here/run_clc_cv.slurm"
