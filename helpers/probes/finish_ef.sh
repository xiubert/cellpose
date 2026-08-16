#!/bin/bash
# Watcher for arms E and F.
#
#  arm D (done)  clc_all_full3_20260814   98 imgs, v4 split (both degeneration animals in fold 3)
#  arm E         clc_all_full4_20260814   99 imgs, v5 split = v4 + the 20x image (job 3479459)
#  arm F         clc_all_rebal_20260814   98 imgs, v6 split = v4 but 8483 pinned to fold 0 (job 3479460)
#  control       clc_all_full_20260604_fold<k>, scored in run_logs/..._v4eval (+ top-ups, job 3479472)
set -uo pipefail
cd "$HOME/cellpose"
module use ~/modulefiles; module load cellpose_env 2>/dev/null

MD=/ix1/pcody/cellpose/models
SUM=run_logs/ef_summary.txt
rm -f run_logs/ef.READY
echo "==== waiting for 3479459 (E) + 3479460 (F) + 3479472 (control top-up) — $(date) ====" | tee "$SUM"
while squeue -M gpu -u "$USER" 2>/dev/null | grep -qE "3479459|3479460|3479472"; do sleep 60; done
echo "all gone $(date)" | tee -a "$SUM"

for k in 0 1 2 3 4; do
    printf "  fold %s: E model %s csv %s | F model %s csv %s\n" "$k" \
      "$([ -f "$MD/clc_all_full4_20260814_fold$k" ] && echo OK || echo MISS)" \
      "$([ -f run_logs/clc_all_full4_20260814/eval_fold$k.csv ] && echo OK || echo MISS)" \
      "$([ -f "$MD/clc_all_rebal_20260814_fold$k" ] && echo OK || echo MISS)" \
      "$([ -f run_logs/clc_all_rebal_20260814/eval_fold$k.csv ] && echo OK || echo MISS)" | tee -a "$SUM"
done

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
            ap[r["image"]] = (float(r["AP@0.5"]), float(r["n_true"]), float(r["n_pred"]))
    return ap

def csv1(p):
    out = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            out[r["image"]] = (float(r["AP@0.5"]), float(r["n_true"]), float(r["n_pred"]))
    return out

A = pool("run_logs/clc_all_full_20260604_v4eval")          # control on the 98
D = pool("run_logs/clc_all_full3_20260814")                # arm D
E = pool("run_logs/clc_all_full4_20260814")                # arm E (99, 20x in)
F = pool("run_logs/clc_all_rebal_20260814")                # arm F (rebalanced)
A20 = csv1("run_logs/ctrl_fold3_on_20x.csv")               # control on the 20x
A8483 = csv1("run_logs/ctrl_fold0_on_8483.csv")            # control on 8483 under fold 0
dep20 = csv1("run_logs/deploy_on_20x.csv")

def cmp(label, X, Y, ks, xn="X", yn="Y"):
    ks = [k for k in ks if k in X and k in Y]
    if not ks:
        print(f"  {label}: no overlap"); return
    x = np.array([X[k][0] for k in ks]); y = np.array([Y[k][0] for k in ks])
    d = x - y
    line = (f"  {label:<44s} n={len(ks):3d}  {xn}={x.mean():.4f} {yn}={y.mean():.4f} "
            f"Δ={d.mean():+.4f} W/T/L={int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())}")
    if wilcoxon is not None and np.any(d != 0):
        try: line += f" p={wilcoxon(x, y)[1]:.4g}"
        except Exception: pass
    print(line)

is20 = lambda s: "20x" in s
sparse = lambda s: ("8477" in s) or ("8483" in s)
a8477 = lambda s: "8477" in s
a8483 = lambda s: "8483" in s

print("\n================ TEST 1: does including the 20x image help or hurt? ================")
print("(the 20x sits in fold 3, so no arm trains on 20x AND tests on it; what this")
print(" measures is the effect of a 20x image in the training pool on the 63x set)")
shared = [k for k in E if not is20(k)]
print("\n-- arm E (99, 20x in training) vs arm D (98, no 20x), same 98 test images:")
cmp("ALL 98 (63x)", E, D, shared, "E", "D")
print("\n-- arm E vs control:")
cmp("ALL 98 (63x)", E, A, shared, "E", "ctrl")
print("\n-- the 20x image itself (held out in fold 3 for every arm):")
for nm, M in (("deployed", dep20), ("control f3", A20), ("arm D (no 20x anywhere)", D), ("arm E (20x in dataset)", E)):
    kk = [k for k in M if is20(k)]
    if kk:
        ap, nt, npd = M[kk[0]]
        print(f"    {nm:<26s} AP@0.5={ap:.3f}  n_true={int(nt)}  n_pred={int(npd)}")

print("\n================ TEST 2: better hold-out for the missing-cells case ================")
print("arm F pins 8483 to fold 0, so each degeneration animal is tested by a model")
print("that TRAINED on the other one. arm D had both in fold 3 (neither in training).")
print("\n-- THE CLEAN TRANSFER TEST: 8477's images, same fold 3 in both arms,")
print("   arm F's fold-3 model additionally trained on 8483's 3 degeneration images:")
for k in sorted([k for k in D if a8477(k)]):
    dd, nt, npd_d = D[k]; ff, _, npd_f = F[k]
    print(f"    GT={int(nt):3d}  armD={dd:.3f} (n_pred={int(npd_d):3d})  ->  armF={ff:.3f} (n_pred={int(npd_f):3d})   {k[42:76]}")
cmp("8477 (3 imgs) F vs D", F, D, [k for k in D if a8477(k)], "F", "D")
print("\n-- 8483's images (arm D: fold-3 model, no degeneration in training;")
print("   arm F: fold-0 model, trained WITH 8477):")
for k in sorted([k for k in D if a8483(k)]):
    dd, nt, npd_d = D[k]; ff, _, npd_f = F[k]
    ctrl = A8483.get(k, (float('nan'),0,0))[0]
    print(f"    GT={int(nt):3d}  ctrl={ctrl:.3f}  armD={dd:.3f} (n_pred={int(npd_d):3d})  ->  armF={ff:.3f} (n_pred={int(npd_f):3d})   {k[42:76]}")
cmp("8483 (3 imgs) F vs D", F, D, [k for k in D if a8483(k)], "F", "D")
print()
cmp("all 6 degeneration imgs, F vs D", F, D, [k for k in D if sparse(k)], "F", "D")
cmp("non-degeneration 92, F vs D", F, D, [k for k in D if not sparse(k)], "F", "D")

print("\n-- arm F vs control (control 8483 rescored under fold 0):")
Afix = dict(A); Afix.update(A8483)
cmp("ALL 98", F, Afix, list(F), "F", "ctrl")
cmp("6 degeneration imgs", F, Afix, [k for k in F if sparse(k)], "F", "ctrl")

print("\n================ noise floor ================")
print("fold 3's training set is IDENTICAL between arms D and E (the 20x is held out")
print("in fold 3), so any D-vs-E difference on fold-3 images is pure training noise:")
f3 = [k for k in shared if ("5056L" in k or "5457 " in k or "5L " in k or "8363" in k or a8477(k) or k.startswith("AL "))]
cmp("fold-3 images, E vs D (should be ~0)", E, D, f3, "E", "D")
PY

echo "" | tee -a "$SUM"
echo "==== submitting error-eval for arms E + F ====" | tee -a "$SUM"
MODELS="heldout:$MD/clc_all_rebal_20260814_fold{k}"
EJ=$(MODELS="$MODELS" DATA_DIRS=/ix1/pcody/cellpose/data/CLC_full3/adult,/ix1/pcody/cellpose/data/CLC_full3/neonate \
     MANIFEST=/ix1/pcody/cellpose/data/CLC_full3/clc_folds_kfold_v6.json \
     sbatch --parsable --export=ALL,MODELS,DATA_DIRS,MANIFEST run_erreval.slurm 2>&1)
echo "erreval(F) job: $EJ" | tee -a "$SUM"
EJ2=$(MODELS="heldout:$MD/clc_all_full4_20260814_fold{k}" \
     DATA_DIRS=/ix1/pcody/cellpose/data/CLC_full4/adult,/ix1/pcody/cellpose/data/CLC_full4/neonate \
     MANIFEST=/ix1/pcody/cellpose/data/CLC_full4/clc_folds_kfold_v5.json \
     sbatch --parsable --export=ALL,MODELS,DATA_DIRS,MANIFEST run_erreval.slurm 2>&1)
echo "erreval(E) job: $EJ2" | tee -a "$SUM"
touch run_logs/ef.READY
echo "==== watcher done $(date) ====" | tee -a "$SUM"
