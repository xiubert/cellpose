#!/bin/bash
# Submit the cellpose training job to SLURM.
#
# Reads PITT_EMAIL from .env (gitignored) and passes it to sbatch as
# --mail-user, so no personal info lives in the committed SLURM script.
# SBATCH directives are parsed at submit time and can't read shell vars, so
# the address has to come in on the sbatch command line — hence this wrapper.
#
# Usage:  ./submit.sh            (any extra args are forwarded to sbatch)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$here/.env" ]; then
    echo "error: $here/.env not found." >&2
    echo "       cp $here/.env.example $here/.env  and set PITT_EMAIL" >&2
    exit 1
fi

set -a
# shellcheck disable=SC1091
source "$here/.env"
set +a

: "${PITT_EMAIL:?set PITT_EMAIL in .env}"

sbatch --mail-user="$PITT_EMAIL" "$here/run_trainer.slurm" "$@"
