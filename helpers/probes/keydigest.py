"""Per-key digest of every seg dict in a directory, so two snapshots can be
compared key-by-key (not just on the masks array)."""
import glob, hashlib, json, os, sys
import numpy as np


def dig(v):
    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape}:{hashlib.md5(np.ascontiguousarray(v)).hexdigest()[:12]}"
    try:
        return "obj:" + hashlib.md5(repr(v).encode()).hexdigest()[:12]
    except Exception:
        return "unhashable"


root = sys.argv[1]
out = {}
for p in sorted(glob.glob(os.path.join(root, "*_seg.npy"))):
    st = os.path.basename(p)[:-len("_seg.npy")]
    d = np.load(p, allow_pickle=True).item()
    out[st] = {k: dig(v) for k, v in d.items()}
print(json.dumps(out))
