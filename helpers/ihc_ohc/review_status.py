"""Record, per new CLC image, how much of it a human actually labelled.

The sidecar cannot distinguish "curator reviewed it and agreed" from "never
opened". Per the 2026-08-14 decision the zero-correction images are treated as
REVIEWED (their class_map_fused counts as validated GT for training), but that
is an assumption, so it is recorded here per image and can be revoked.
"""
import glob, json, os, datetime
import numpy as np

out = {"recorded": datetime.date.today().isoformat(),
       "assumption": "images with zero class_map_user are treated as reviewed-and-agreed "
                     "(class_map_fused = validated GT). Revocable: see 'risk' per image.",
       "images": {}}
for d in ["/data/cellpose_cc/adult(2)", "/data/cellpose_cc/neonate(2)"]:
    for p in sorted(glob.glob(os.path.join(d, "*_pred.npy"))):
        st = os.path.basename(p)[:-len("_pred.npy")]
        o = np.load(p, allow_pickle=True).item()
        u = o.get("class_map_user") or {}
        f = o.get("class_map_fused") or {}
        seg = p[:-len("_pred.npy")] + "_seg.npy"
        nm = int(np.load(seg, allow_pickle=True).item()["masks"].max()) if os.path.exists(seg) else 0
        diff = sum(1 for k, v in u.items() if str(f.get(k)) != str(v))
        frac = len(u) / nm if nm else 0.0
        sparse = nm < 80
        is20x = "20x" in st
        if u:
            status = "corrected"
        else:
            status = "assumed_reviewed"
        # riskiest assumption: no corrections on an image type where the model is
        # known to be wrong (sparse degeneration, or 20x)
        risk = "high" if (status == "assumed_reviewed" and (sparse or is20x)) else \
               ("none" if status == "corrected" else "normal")
        out["images"][st] = {
            "dir": os.path.basename(d), "masks": nm, "user_labeled": len(u),
            "corrections": diff, "frac_labeled": round(frac, 3),
            "status": status, "risk": risk,
            "sparse": bool(sparse), "magnification": "20x" if is20x else "63x",
        }
print(json.dumps(out, indent=1))
