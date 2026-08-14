#!/bin/bash
# Watcher for the "does the 2026-07 batch (+20 images) help?" experiment.
#
# Arm A (control): clc_all_full_20260604_fold<k>  — trained on the old 67 only
#                  (job 3346061 re-scores it on the EXTENDED fold test sets)
# Arm B (test)   : clc_all_full2_20260731_fold<k> — same recipe, trained on all 87
#                  (array job 3346056)
# Both are scored on identical held-out images (clc_split.py --extend pinned every
# old animal to its original fold), so every comparison below is paired.
#
# When the array finishes: pool paired AP@0.5, break out old-vs-new / age / the
# 20x outlier, sanity-check the harness against the published control CSVs, then
# submit the curation error-eval (FN-recovery / FP-suppression / TP-retention).
set -uo pipefail
cd "$HOME/cellpose"
module use ~/modulefiles; module load cellpose_env 2>/dev/null

ARRAY=3346056
CTRL_JOB=3346061
MD=/ix1/pcody/cellpose/models
MAN=/ix1/pcody/cellpose/data/CLC_full2/clc_folds_kfold_v2.json
NEW_RUN=run_logs/clc_all_full2_20260731
CTRL_RUN=run_logs/clc_all_full_20260604_v2eval
OLD_RUN=run_logs/clc_all_full_20260604
SUM=run_logs/full2_summary.txt

rm -f run_logs/full2.READY
echo "==== waiting for $ARRAY (arm B) + $CTRL_JOB (arm A eval) — $(date) ====" | tee "$SUM"
while squeue -M gpu -u "$USER" 2>/dev/null | grep -qE "$ARRAY|$CTRL_JOB"; do sleep 60; done
echo "both gone $(date)" | tee -a "$SUM"

echo "" | tee -a "$SUM"
echo "==== artifacts ====" | tee -a "$SUM"
for k in 0 1 2 3 4; do
    m="$MD/clc_all_full2_20260731_fold$k"
    printf "  fold %s: model %s  armB csv %s  armA csv %s\n" "$k" \
        "$([ -f "$m" ] && echo OK || echo MISSING)" \
        "$([ -f "$NEW_RUN/eval_fold$k.csv" ] && echo OK || echo MISSING)" \
        "$([ -f "$CTRL_RUN/eval_fold$k.csv" ] && echo OK || echo MISSING)" | tee -a "$SUM"
done

# one CSV for the control arm (aggregate --baseline_csv takes a single file)
CTRL_CSV=run_logs/ctrl_full2_all.csv
awk 'FNR==1 && NR!=1 {next} {print}' $CTRL_RUN/eval_fold*.csv > "$CTRL_CSV" 2>/dev/null

echo "" | tee -a "$SUM"
echo "==== paired CV AP (arm B vs arm A, same 87 held-out images) ====" | tee -a "$SUM"
python clc_cv.py aggregate --manifest "$MAN" --results_dir "$NEW_RUN" \
    --baseline_csv "$CTRL_CSV" 2>&1 | tee -a "$SUM"

echo "" | tee -a "$SUM"
echo "==== breakdown: old-67 vs new-20, and the 20x outlier ====" | tee -a "$SUM"
python - "$NEW_RUN" "$CTRL_RUN" "$OLD_RUN" <<'PY' 2>&1 | tee -a "$SUM"
import csv, glob, os, sys
import numpy as np
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

new_dir, ctrl_dir, old_dir = sys.argv[1:4]
NEW_STEMS = set()          # the 2026-07 batch, by animal
NEW_ANIMALS = {"8363", "AL", "BL", "CL", "DL"}

def pool(d):
    ap = {}
    for f in sorted(glob.glob(os.path.join(d, "eval_fold*.csv"))):
        for r in csv.DictReader(f and open(f)):
            ap[r["image"]] = {c: float(r[c]) for c in r if c.startswith("AP@")}
    return ap

B, A, OLD = pool(new_dir), pool(ctrl_dir), pool(old_dir)
keys = sorted(set(A) & set(B))
print(f"paired images: {len(keys)}  (armA {len(A)}, armB {len(B)})")

# harness sanity: control re-score must reproduce the published control CSVs
same = [k for k in OLD if k in A]
if same:
    d = np.array([A[k]["AP@0.5"] - OLD[k]["AP@0.5"] for k in same])
    print(f"harness check: {len(same)} old images re-scored, max|Δ AP@0.5| = {np.abs(d).max():.6f}"
          f"  ({'REPRODUCES' if np.abs(d).max() < 1e-6 else 'MISMATCH — investigate'})")

def is_new(stem):
    # the new batch: 5 brand-new animals + the three re-imaged 4L files
    return any(stem.startswith(a + " ") for a in NEW_ANIMALS) or "8363" in stem \
        or stem.startswith("4L ")

def report(label, ks):
    if not ks:
        return
    for m in ("AP@0.5", "AP@0.75", "AP@0.9"):
        b = np.array([B[k][m] for k in ks]); a = np.array([A[k][m] for k in ks])
        d = b - a
        line = (f"  {label:<22s} {m:<8s} armB={b.mean():.4f} armA={a.mean():.4f} "
                f"Δ={d.mean():+.4f} W/T/L={int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())} n={len(ks)}")
        if wilcoxon is not None and np.any(d != 0) and m == "AP@0.5":
            try:
                line += f" p={wilcoxon(b, a)[1]:.4g}"
            except Exception:
                pass
        print(line)
    print()

report("ALL", keys)
report("old 67 only", [k for k in keys if not is_new(k)])
report("new batch only", [k for k in keys if is_new(k)])
x20 = [k for k in keys if "20x" in k]
for k in x20:
    print(f"  20x outlier: {k[:60]}  armA={A[k]['AP@0.5']:.3f} armB={B[k]['AP@0.5']:.3f}")
report("ALL minus 20x", [k for k in keys if k not in set(x20)])

worst = sorted(keys, key=lambda k: B[k]["AP@0.5"] - A[k]["AP@0.5"])
print("biggest per-image swings (Δ AP@0.5 = armB - armA):")
for k in worst[:5] + worst[-5:]:
    print(f"   {B[k]['AP@0.5']-A[k]['AP@0.5']:+.3f}  {k[:66]}")
PY

echo "" | tee -a "$SUM"
echo "==== submitting curation error-eval for BOTH arms (87 images) ====" | tee -a "$SUM"
MODELS="cpsam,heldout:$MD/clc_all_full_20260604_fold{k},heldout:$MD/clc_all_full2_20260731_fold{k}"
DATA_DIRS=/ix1/pcody/cellpose/data/CLC_full2/adult,/ix1/pcody/cellpose/data/CLC_full2/neonate
EJ=$(MODELS="$MODELS" DATA_DIRS="$DATA_DIRS" MANIFEST="$MAN" \
     sbatch --parsable --export=ALL,MODELS,DATA_DIRS,MANIFEST run_erreval.slurm 2>&1)
echo "erreval job: $EJ" | tee -a "$SUM"
touch run_logs/full2.READY
echo "==== watcher done $(date) ====" | tee -a "$SUM"
