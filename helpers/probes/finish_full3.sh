#!/bin/bash
# Watcher for the 2026-08 batch experiment (CLC_full3, 98 images).
#
# Arm A' (control): clc_all_full_20260604_fold<k>  — the deployed model's leak-free
#                   CV twin, RE-SCORED against current GT (job 3479318)
# Arm D  (test)   : clc_all_full3_20260814_fold<k> — same recipe, trained on all 98
#                   (job 3479313)
# Plus: clc_all_full_deploy scored directly on the 31 images it has never seen.
#
# Folds were pinned with clc_split.py --extend, so every old animal sits in the
# same fold as in all previous rounds and the comparison is paired per image.
set -uo pipefail
cd "$HOME/cellpose"
module use ~/modulefiles; module load cellpose_env 2>/dev/null

ARRAY=3479313
CTRL=3479318
MD=/ix1/pcody/cellpose/models
MAN=/ix1/pcody/cellpose/data/CLC_full3/clc_folds_kfold_v4.json
SUM=run_logs/full3_summary.txt

rm -f run_logs/full3.READY
echo "==== waiting for $ARRAY (arm D) + $CTRL (control) — $(date) ====" | tee "$SUM"
while squeue -M gpu -u "$USER" 2>/dev/null | grep -qE "$ARRAY|$CTRL"; do sleep 60; done
echo "both gone $(date)" | tee -a "$SUM"

echo "" | tee -a "$SUM"
for k in 0 1 2 3 4; do
    printf "  fold %s: model %s  armD csv %s  armA' csv %s\n" "$k" \
        "$([ -f "$MD/clc_all_full3_20260814_fold$k" ] && echo OK || echo MISSING)" \
        "$([ -f run_logs/clc_all_full3_20260814/eval_fold$k.csv ] && echo OK || echo MISSING)" \
        "$([ -f run_logs/clc_all_full_20260604_v4eval/eval_fold$k.csv ] && echo OK || echo MISSING)" | tee -a "$SUM"
done

echo "" | tee -a "$SUM"
echo "==== paired AP: arm D (98 imgs) vs arm A' (67 imgs), same held-out images ====" | tee -a "$SUM"
python - <<'PY' 2>&1 | tee -a "$SUM"
import csv, glob, os
import numpy as np
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

def pool(d):
    ap = {}
    for f in sorted(glob.glob(os.path.join(d, "eval_fold*.csv"))):
        for r in csv.DictReader(open(f)):
            ap[r["image"]] = {c: float(r[c]) for c in r if c.startswith("AP@")}
            ap[r["image"]]["n_true"] = float(r.get("n_true", 0) or 0)
    return ap

A = pool("run_logs/clc_all_full_20260604_v4eval")
D = pool("run_logs/clc_all_full3_20260814")
keys = sorted(set(A) & set(D))
print(f"paired images: {len(keys)}  (armA' {len(A)}, armD {len(D)})\n")

def is_old(s):    return "CellPose Quant" in s
def is_aug26(s):  return "tomt overexp" in s                      # 2026-08 batch
def is_sparse(s): return ("8477" in s) or ("8483" in s)           # degeneration phenotype
def is_jul(s):    return (not is_old(s)) and (not is_aug26(s))    # 2026-07 batch

def block(label, ks):
    if not ks:
        return
    print(f"--- {label}  (n={len(ks)}, {int(sum(A[k]['n_true'] for k in ks))} GT cells)")
    for m in ("AP@0.5", "AP@0.75", "AP@0.9"):
        a = np.array([A[k][m] for k in ks]); d = np.array([D[k][m] for k in ks])
        line = f"    {m:<8s} armA'={a.mean():.4f}  armD={d.mean():.4f}  Δ={(d-a).mean():+.4f}"
        if m == "AP@0.5":
            df = d - a
            line += f"  W/T/L={int((df>0).sum())}/{int((df==0).sum())}/{int((df<0).sum())}"
            if wilcoxon is not None and np.any(df != 0):
                try:
                    line += f"  p={wilcoxon(d, a)[1]:.4g}"
                except Exception:
                    pass
        print(line)
    print()

block("ALL", keys)
block("old 67", [k for k in keys if is_old(k)])
block("2026-07 batch", [k for k in keys if is_jul(k)])
block("2026-08 batch", [k for k in keys if is_aug26(k)])
block("  ..of which sparse (8477/8483)", [k for k in keys if is_sparse(k)])
block("ALL minus sparse", [k for k in keys if not is_sparse(k)])

sw = sorted(keys, key=lambda k: D[k]["AP@0.5"] - A[k]["AP@0.5"])
print("biggest per-image swings (Δ AP@0.5 = armD - armA'):")
for k in sw[:6] + sw[-6:]:
    print(f"   {D[k]['AP@0.5']-A[k]['AP@0.5']:+.3f}  (n_true={int(A[k]['n_true']):4d})  {k[:58]}")

# what the SHIPPED model does on images it has never seen
p = "run_logs/deploy_on_newonly.csv"
if os.path.exists(p):
    rows = list(csv.DictReader(open(p)))
    v = np.array([float(r["AP@0.5"]) for r in rows])
    print(f"\nclc_all_full_deploy on the {len(v)} never-seen new images: "
          f"mean AP@0.5 = {v.mean():.4f}  median = {np.median(v):.4f}  min = {v.min():.3f}")
    for r in sorted(rows, key=lambda r: float(r["AP@0.5"]))[:6]:
        print(f"    {float(r['AP@0.5']):.3f}  {r['image'][:58]}")
PY

echo "" | tee -a "$SUM"
echo "==== submitting curation error-eval (cpsam + arm A' + arm D) on 98 images ====" | tee -a "$SUM"
MODELS="cpsam,heldout:$MD/clc_all_full_20260604_fold{k},heldout:$MD/clc_all_full3_20260814_fold{k}"
DATA_DIRS=/ix1/pcody/cellpose/data/CLC_full3/adult,/ix1/pcody/cellpose/data/CLC_full3/neonate
EJ=$(MODELS="$MODELS" DATA_DIRS="$DATA_DIRS" MANIFEST="$MAN" \
     sbatch --parsable --export=ALL,MODELS,DATA_DIRS,MANIFEST run_erreval.slurm 2>&1)
echo "erreval job: $EJ" | tee -a "$SUM"
touch run_logs/full3.READY
echo "==== watcher done $(date) ====" | tee -a "$SUM"
