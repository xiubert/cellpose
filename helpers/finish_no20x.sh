#!/bin/bash
# Watcher for arm C — "does dropping the single 20x acquisition from TRAINING
# unlock the gain the 2026-07 batch didn't deliver?"
#
# Arm A (control): clc_all_full_20260604_fold<k>       trained on old 67
# Arm B          : clc_all_full2_20260731_fold<k>      trained on all 87 (incl 20x)
# Arm C (this)   : clc_all_full2_no20x_20260731_fold<k> trained on 86 (20x dropped)
#
# All three are scored on the SAME 86 images (the 20x image is excluded from the
# comparison entirely — it is out of domain and would be its own training target).
# Arm A/B per-image CSVs already exist, so this only has to wait for arm C.
set -uo pipefail
cd "$HOME/cellpose"
module use ~/modulefiles; module load cellpose_env 2>/dev/null

ARRAY=3346149
MD=/ix1/pcody/cellpose/models
MAN=/ix1/pcody/cellpose/data/CLC_full2_no20x/clc_folds_kfold_v3.json
SUM=run_logs/no20x_summary.txt

rm -f run_logs/no20x.READY
echo "==== waiting for $ARRAY (arm C) — $(date) ====" | tee "$SUM"
while squeue -M gpu -u "$USER" 2>/dev/null | grep -q "$ARRAY"; do sleep 60; done
echo "array gone $(date)" | tee -a "$SUM"

echo "" | tee -a "$SUM"
for k in 0 1 2 3 4; do
    printf "  fold %s: model %s  csv %s\n" "$k" \
        "$([ -f "$MD/clc_all_full2_no20x_20260731_fold$k" ] && echo OK || echo MISSING)" \
        "$([ -f run_logs/clc_all_full2_no20x_20260731/eval_fold$k.csv ] && echo OK || echo MISSING)" | tee -a "$SUM"
done

echo "" | tee -a "$SUM"
echo "==== three-arm paired comparison on the SAME 86 images ====" | tee -a "$SUM"
python - <<'PY' 2>&1 | tee -a "$SUM"
import csv, glob, os
import numpy as np
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

ARMS = {
    "A: 67 imgs (deployed recipe)": "run_logs/clc_all_full_20260604_v2eval",
    "B: 87 imgs (incl 20x)":        "run_logs/clc_all_full2_20260731",
    "C: 86 imgs (20x dropped)":     "run_logs/clc_all_full2_no20x_20260731",
}

def pool(d):
    ap = {}
    for f in sorted(glob.glob(os.path.join(d, "eval_fold*.csv"))):
        for r in csv.DictReader(open(f)):
            ap[r["image"]] = {c: float(r[c]) for c in r if c.startswith("AP@")}
    return ap

P = {k: pool(v) for k, v in ARMS.items()}
for k, v in P.items():
    print(f"  {k:<32s} {len(v)} scored images")

keys = sorted(set.intersection(*[set(v) for v in P.values()]))
keys = [k for k in keys if "20x" not in k]          # the 20x image is out of scope
print(f"\ncommon non-20x images: {len(keys)}\n")

NEW = ("AL ", "BL ", "CL ", "DL ", "4L ")
def is_new(s):
    return s.startswith(NEW) or "8363" in s

def block(label, ks):
    if not ks:
        return
    print(f"--- {label}  (n={len(ks)})")
    for m in ("AP@0.5", "AP@0.75", "AP@0.9"):
        vals = {a: np.array([P[a][k][m] for k in ks]) for a in ARMS}
        print("    " + m + "  " + "  ".join(f"{a.split(':')[0]}={v.mean():.4f}" for a, v in vals.items()))
    a = np.array([P["A: 67 imgs (deployed recipe)"][k]["AP@0.5"] for k in ks])
    for other in ("B: 87 imgs (incl 20x)", "C: 86 imgs (20x dropped)"):
        b = np.array([P[other][k]["AP@0.5"] for k in ks])
        d = b - a
        line = (f"    {other.split(':')[0]} vs A  AP@0.5 Δ={d.mean():+.4f} "
                f"W/T/L={int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())}")
        if wilcoxon is not None and np.any(d != 0):
            try:
                line += f"  p={wilcoxon(b, a)[1]:.4g}"
            except Exception:
                pass
        print(line)
    b = np.array([P["B: 87 imgs (incl 20x)"][k]["AP@0.5"] for k in ks])
    c = np.array([P["C: 86 imgs (20x dropped)"][k]["AP@0.5"] for k in ks])
    d = c - b
    line = (f"    C vs B  AP@0.5 Δ={d.mean():+.4f} "
            f"W/T/L={int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())}")
    if wilcoxon is not None and np.any(d != 0):
        try:
            line += f"  p={wilcoxon(c, b)[1]:.4g}"
        except Exception:
            pass
    print(line + "\n")

block("ALL non-20x", keys)
block("old 67", [k for k in keys if not is_new(k)])
block("new batch (19, 20x excluded)", [k for k in keys if is_new(k)])

A = "A: 67 imgs (deployed recipe)"; C = "C: 86 imgs (20x dropped)"
sw = sorted(keys, key=lambda k: P[C][k]["AP@0.5"] - P[A][k]["AP@0.5"])
print("biggest per-image swings (Δ AP@0.5 = armC - armA):")
for k in sw[:5] + sw[-5:]:
    print(f"   {P[C][k]['AP@0.5']-P[A][k]['AP@0.5']:+.3f}  {k[:66]}")
PY

echo "" | tee -a "$SUM"
echo "==== submitting error-eval for ALL THREE arms on the same 86 images ====" | tee -a "$SUM"
# All three, because dropping the 20x image changes the TP/FP denominators (it
# alone carries 342 cells, none hand-drawn) — the 87-image numbers from job
# 3346128 are not comparable to an 86-image run.
MODELS="cpsam,heldout:$MD/clc_all_full_20260604_fold{k},heldout:$MD/clc_all_full2_20260731_fold{k},heldout:$MD/clc_all_full2_no20x_20260731_fold{k}"
DATA_DIRS=/ix1/pcody/cellpose/data/CLC_full2_no20x/adult,/ix1/pcody/cellpose/data/CLC_full2_no20x/neonate
EJ=$(MODELS="$MODELS" DATA_DIRS="$DATA_DIRS" MANIFEST="$MAN" \
     sbatch --parsable --export=ALL,MODELS,DATA_DIRS,MANIFEST run_erreval.slurm 2>&1)
echo "erreval job: $EJ" | tee -a "$SUM"
touch run_logs/no20x.READY
echo "==== watcher done $(date) ====" | tee -a "$SUM"
