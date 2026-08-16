"""Empirical training-noise floor on the degeneration images.

Arms D and E have IDENTICAL fold-3 training data (the 20x image is held out in
fold 3, so adding it to the dataset changes nothing about what fold 3 trains on),
and 8477+8483 both live in fold 3 under v4/v5. So D-vs-E on those six images is
two runs of the same recipe on the same data: pure run-to-run variance.
Any 'effect' smaller than this is not an effect.
"""
import csv, glob, os
import numpy as np

def pool(d):
    ap = {}
    for f in sorted(glob.glob(os.path.join(d, "eval_fold*.csv"))):
        for r in csv.DictReader(open(f)):
            ap[r["image"]] = (float(r["AP@0.5"]), float(r["n_true"]), float(r["n_pred"]))
    return ap

D = pool("run_logs/clc_all_full3_20260814")
E = pool("run_logs/clc_all_full4_20260814")
F = pool("run_logs/clc_all_rebal_20260814")
sparse = sorted([k for k in D if ("8477" in k) or ("8483" in k)])

print("SAME-DATA REPLICATE (arm D vs arm E) on the 6 degeneration images:")
print(f"{'GT':>5} {'armD':>7} {'armE':>7} {'|diff|':>7}   image")
dif = []
for k in sparse:
    d, nt, npd = D[k]; e, _, npe = E[k]
    dif.append(abs(d - e))
    print(f"{int(nt):5d} {d:7.3f} {e:7.3f} {abs(d-e):7.3f}   {k[42:74]}")
dif = np.array(dif)
print(f"\n  mean |D-E| on these 6 images = {dif.mean():.3f}   max = {dif.max():.3f}")
dD = np.array([D[k][0] for k in sparse]); dE = np.array([E[k][0] for k in sparse])
print(f"  subset means: D={dD.mean():.4f}  E={dE.mean():.4f}  -> replicate Δ = {dE.mean()-dD.mean():+.4f}")
dF = np.array([F[k][0] for k in sparse])
print(f"\n  arm F (rebalanced hold-out) on the same 6 = {dF.mean():.4f}")
print(f"  F - D = {dF.mean()-dD.mean():+.4f}   vs a same-data replicate gap of {abs(dE.mean()-dD.mean()):.4f}")
allk = [k for k in D if k in E]
aD = np.array([D[k][0] for k in allk]); aE = np.array([E[k][0] for k in allk])
print(f"\n  for scale, same-data replicate on all {len(allk)} images: D={aD.mean():.4f} E={aE.mean():.4f} Δ={aE.mean()-aD.mean():+.4f}")
