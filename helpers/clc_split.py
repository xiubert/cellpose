"""
Leak-free GroupKFold split of the in-house CLC cochlear dataset.

WHY GROUP BY ANIMAL (not image)
-------------------------------
CLC filenames encode <animal> <freq>khz, and several tonotopic frequency
regions come from the SAME cochlea/animal, e.g. 5042L 8khz / 16khz / 32khz.
Two regions of one cochlea share tissue, staining batch and animal, so they
must never straddle train/test — the split groups by **animal ID**, not by
the individual image. (With the D4 offline augments now deleted there is one
original image per <animal,freq>; grouping is still by animal.)

WHAT IT DOES
------------
Scans CLC subdirs (default adult/ + neonate/), parses (animal, age) per
*_seg.npy, builds folds that keep every animal's images together, and writes a
JSON manifest of train/test stem lists per fold. Originals only (skips any
residual _SV_<tag>/_rot/_flip augmented stems). No pixels needed — pure
filename logic, numpy-only (no sklearn), so it runs in the bare cellpose env.

Schemes:
  loao   (default) leave-one-animal-out — one fold per animal (test = that
         animal's 1-3 images, train = all others). Best for very small grouped
         data: max train per fold, every animal tested once.
  kfold  --n_folds K, age-stratified: adult and neonate animals are each
         round-robined across folds so every fold's test set mixes ages.

Usage:
  python clc_split.py --root /ix1/pcody/cellpose/data/CLC \
      [--scheme loao|kfold] [--n_folds 4] [--out clc_folds.json] [--seed 42]
"""

import argparse
import glob
import json
import os
import re

import numpy as np

_AUG_MARKERS = ("_SV_", "_rot90", "_rot180", "_rot270", "_fliph", "_flipv")
# animal id token, either
#   numeric  : digits optionally followed by L/R  (5042L, 5165, 1L, 2L, 8363)
#   lettered : 1-3 capitals ending in L/R         (AL, BL, CL, DL — the 2026-07 batch)
# The lettered form needs >=2 chars so a stray "L" token can't match, and the
# trailing L/R keeps it from matching things like "HET" or "SV".
_ANIMAL_RE = re.compile(r"^(?:\d+[LR]?|[A-Z]{1,2}[LR])$")
# frequency anchor: "8khz", "16 khz", "32khz " ...
_FREQ_RE = re.compile(r"(\d+)\s*khz", re.IGNORECASE)


def seg_stem(p):
    b = os.path.basename(p)
    return b[:-len("_seg.npy")] if b.endswith("_seg.npy") else os.path.splitext(b)[0]


def is_aug(stem):
    return any(m in stem for m in _AUG_MARKERS)


def parse_animal(stem):
    """Animal id = the last animal-like token (\\d+[LR]?) before the freq anchor.

    Robust to the irregular naming: 'Samples_10_63x 5042L 8khz' -> 5042L,
    'samples_1_4L 8khz' -> 4L (animal glued into the sample token),
    '3L 8 khz' (space in freq) -> 3L.
    """
    m = _FREQ_RE.search(stem)
    head = stem[:m.start()] if m else stem
    tokens = re.split(r"[\s_]+", head)
    cands = [t for t in tokens if _ANIMAL_RE.match(t)]
    if not cands:
        raise ValueError(f"could not parse animal id from stem: {stem!r}")
    return cands[-1]


def scan(root, subdirs):
    """Return list of dicts: {path, stem, animal, age} for each original seg."""
    items = []
    for age in subdirs:
        d = os.path.join(root, age)
        for p in sorted(glob.glob(os.path.join(d, "*_seg.npy"))):
            st = seg_stem(p)
            if is_aug(st):
                continue
            items.append({"path": p, "stem": st, "animal": parse_animal(st), "age": age})
    return items


def _extend_groups(animals, names, prior_path):
    """Keep a previous manifest's fold assignment; add only the NEW animals.

    WHY: when new labeled data arrives, re-running the split from scratch
    reshuffles every animal, which makes the new CV run un-paired with the
    previous one — you can no longer tell "more data helped" apart from "the
    test sets changed". Extending pins each already-assigned animal to its old
    fold, so previously trained fold models stay valid held-out scorers for the
    same images, and new animals are added where they balance the folds best
    (greedy: biggest animal first -> fold with the fewest images).
    """
    with open(prior_path) as fh:
        prior = json.load(fh)
    groups = [list(f["test_animals"]) for f in prior["folds"]]
    known = {a for g in groups for a in g}
    counts = [sum(len(animals[a]["idx"]) for a in g if a in animals) for g in groups]

    missing = sorted(known - set(names))
    new = sorted(set(names) - known, key=lambda a: (-len(animals[a]["idx"]), a))
    for a in new:
        k = int(np.argmin(counts))
        groups[k].append(a)
        counts[k] += len(animals[a]["idx"])
    return groups, prior, new, missing


def make_folds(items, scheme, n_folds, seed, extend=None):
    # animal -> age, and animal -> list of item indices
    animals = {}
    for i, it in enumerate(items):
        animals.setdefault(it["animal"], {"age": it["age"], "idx": []})["idx"].append(i)
    names = sorted(animals)
    extend_info = None

    if extend:
        groups, prior, new, missing = _extend_groups(animals, names, extend)
        extend_info = {"prior": extend, "prior_folds": prior["n_folds"],
                       "new_animals": new, "missing_from_prior": missing}
        if missing:
            print(f"WARNING: {len(missing)} animal(s) in the prior manifest are absent "
                  f"from this root: {missing}")
        print(f"extending {extend}: {len(names) - len(new)} animals keep their fold, "
              f"{len(new)} new animal(s) assigned: {new}")
    elif scheme == "loao":
        groups = [[a] for a in names]                      # one animal per fold
    else:
        rng = np.random.default_rng(seed)
        groups = [[] for _ in range(n_folds)]
        # round-robin per age so each fold mixes adult+neonate
        for age in sorted({animals[a]["age"] for a in names}):
            ages = [a for a in names if animals[a]["age"] == age]
            rng.shuffle(ages)
            for j, a in enumerate(ages):
                groups[j % n_folds].append(a)

    folds = []
    for k, test_animals in enumerate(groups):
        test_set = set(test_animals)
        test_idx = [i for a in test_animals if a in animals for i in animals[a]["idx"]]
        train_idx = [i for i in range(len(items)) if items[i]["animal"] not in test_set]
        folds.append({
            "fold": k,
            "test_animals": sorted(test_animals),
            "train": [items[i]["stem"] for i in sorted(train_idx)],
            "test":  [items[i]["stem"] for i in sorted(test_idx)],
        })
    return folds, animals, extend_info


def parse_args():
    p = argparse.ArgumentParser(description="Leak-free GroupKFold-by-animal split for CLC")
    p.add_argument("--root", default="/ix1/pcody/cellpose/data/CLC")
    p.add_argument("--subdirs", default="adult,neonate")
    p.add_argument("--scheme", choices=["loao", "kfold"], default="loao")
    p.add_argument("--n_folds", type=int, default=4, help="only for --scheme kfold")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extend", default=None,
                   help="prior manifest JSON: keep its fold assignment for animals it "
                        "already covers and only place NEW animals (keeps a new CV run "
                        "paired with the old one). Overrides --scheme/--seed placement.")
    p.add_argument("--out", default=None, help="JSON manifest path (default: <root>/clc_folds_<scheme>.json)")
    return p.parse_args()


def main():
    args = parse_args()
    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
    items = scan(args.root, subdirs)
    if not items:
        raise SystemExit(f"no original *_seg.npy under {args.root}/{{{','.join(subdirs)}}}")

    folds, animals, extend_info = make_folds(items, args.scheme, args.n_folds,
                                             args.seed, extend=args.extend)

    # --- summary ---
    print(f"CLC split — root={args.root}  scheme={args.scheme}"
          + (f"  n_folds={args.n_folds}" if args.scheme == "kfold" else "")
          + (f"  EXTENDED from {args.extend}" if args.extend else ""))
    print(f"{len(items)} images, {len(animals)} animals\n")
    print("animal     age       n_img")
    print("-" * 32)
    for a in sorted(animals):
        print(f"{a:<10} {animals[a]['age']:<9} {len(animals[a]['idx'])}")
    print()
    print(f"{len(folds)} folds:")
    for f in folds:
        ages = sorted({items[[it['stem'] for it in items].index(s)]['age'] for s in f['test']}) if False else None
        print(f"  fold {f['fold']}: test_animals={f['test_animals']}  "
              f"n_train={len(f['train'])}  n_test={len(f['test'])}")

    out = args.out or os.path.join(args.root, f"clc_folds_{args.scheme}.json")
    manifest = {
        "root": args.root, "subdirs": subdirs, "scheme": args.scheme,
        "n_folds": len(folds), "seed": args.seed,
        "animals": {a: {"age": animals[a]["age"], "n_img": len(animals[a]["idx"])} for a in sorted(animals)},
        "folds": folds,
    }
    if extend_info:
        manifest["extends"] = extend_info
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote manifest: {out}")


if __name__ == "__main__":
    main()
