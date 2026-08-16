#!/usr/bin/env bash
# Train/test split for Cellpose data (paired .tif + _seg.npy files).
#
# Splits at the ORIGINAL SAMPLE level — all augmented variants of a sample
# go to the same split — to prevent leakage between train and test.
# (Ideal workflow is to augment AFTER splitting; this script is the safe
# workaround when augmentation was already run on the full dataset.)
#
# Usage:
#   ./train_test_split.sh <src_dir> <out_dir> [test_ratio] [seed]
#
# Defaults: test_ratio=0.2, seed=42
# Output:   <out_dir>/train/   <out_dir>/test/

set -euo pipefail

SRC="${1:?Usage: $0 <src_dir> <out_dir> [test_ratio] [seed]}"
OUT="${2:?Usage: $0 <src_dir> <out_dir> [test_ratio] [seed]}"
TEST_RATIO="${3:-0.2}"
SEED="${4:-42}"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: source directory not found: $SRC" >&2
    exit 1
fi

mkdir -p "$OUT/train" "$OUT/test"

# Extract unique sample identifiers (everything up to the first augmentation
# suffix, e.g. "_SV_fliph" / "_SV_rot90_flipv" / "_SV").
# Pattern: filename stem before _SV... (the augment script uses _SV as base).
mapfile -t TIFS < <(find "$SRC" -maxdepth 1 -name "*.tif" | sort)

if [[ ${#TIFS[@]} -eq 0 ]]; then
    echo "ERROR: no .tif files found in $SRC" >&2
    exit 1
fi

# Build list of unique sample bases (strip _SV* suffix + .tif extension).
declare -A SEEN_BASES
BASES=()
for f in "${TIFS[@]}"; do
    fname="$(basename "$f")"
    # Remove extension then strip _SV and everything after it
    stem="${fname%.tif}"
    base="${stem%%_SV*}"
    if [[ -z "${SEEN_BASES[$base]+x}" ]]; then
        SEEN_BASES["$base"]=1
        BASES+=("$base")
    fi
done

N_TOTAL=${#BASES[@]}
N_TEST=$(awk "BEGIN{printf \"%d\", $N_TOTAL * $TEST_RATIO + 0.5}")
N_TRAIN=$(( N_TOTAL - N_TEST ))

echo "Total original samples : $N_TOTAL"
echo "Train                  : $N_TRAIN"
echo "Test                   : $N_TEST"
echo "Test ratio             : $TEST_RATIO"
echo "Random seed            : $SEED"
echo ""

# Shuffle deterministically using awk seeded RNG, then split.
mapfile -t SHUFFLED < <(
    printf '%s\n' "${BASES[@]}" | \
    awk -v seed="$SEED" 'BEGIN{srand(seed)} {lines[NR]=$0}
     END{
       for(i=NR;i>1;i--){j=int(rand()*i)+1; t=lines[i]; lines[i]=lines[j]; lines[j]=t}
       for(i=1;i<=NR;i++) print lines[i]
     }'
)

TEST_BASES=("${SHUFFLED[@]:0:$N_TEST}")
TRAIN_BASES=("${SHUFFLED[@]:$N_TEST}")

copy_sample() {
    local base="$1"
    local dest="$2"
    local count=0
    # Match all files whose name starts with this base and ends in .tif or _seg.npy
    while IFS= read -r -d '' f; do
        fname="$(basename "$f")"
        cp "$f" "$dest/$fname"
        (( count++ ))
    done < <(find "$SRC" -maxdepth 1 \( -name "${base}_SV*.tif" -o -name "${base}_SV*_seg.npy" \) -print0)
    echo "$count"
}

echo "Copying test samples..."
for base in "${TEST_BASES[@]}"; do
    n=$(copy_sample "$base" "$OUT/test")
    echo "  [test]  $base  ($n files)"
done

echo "Copying train samples..."
for base in "${TRAIN_BASES[@]}"; do
    n=$(copy_sample "$base" "$OUT/train")
    echo "  [train] $base  ($n files)"
done

echo ""
echo "Done."
echo "  Train files : $(find "$OUT/train" -name '*.tif' | wc -l) images, $(find "$OUT/train" -name '*_seg.npy' | wc -l) masks"
echo "  Test files  : $(find "$OUT/test"  -name '*.tif' | wc -l) images, $(find "$OUT/test"  -name '*_seg.npy' | wc -l) masks"
