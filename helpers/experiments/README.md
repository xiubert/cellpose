# Settled experiment arms — do not re-run

Training configs for hypotheses that were **tested and returned no improvement**.
They live here rather than beside `trainer.yaml` / `trainer_deploy.yaml` so the
top level holds only the active pipeline: the baseline recipe and the production
recipe.

Each was run as a full grouped 5-fold CV, paired against the same control on the
same held-out images. Kept because a measured negative is a result — it stops the
next person spending a week rediscovering it — and because the configs document
exactly what was tried.

| config | hypothesis | result | verdict |
|---|---|---|---|
| `trainer_warmstart.yaml` | Initialising from the Cunningham model (`label_xfer_aug_retest`) and refining gently beats training from stock cpsam | AP@0.5 **0.827** vs 0.834 from-cpsam, Δ−0.006, **p=0.75** (job 2647218) | **Tied.** Rejected — it adds a dependency on the Cunningham model for no gain |
| `trainer_fnaug.yaml` | Cutout and blur oppose *detection*, so dropping them should recover false negatives | FN-recovery **72.1%** vs 71.9% baseline; AP flat (job 2701806) | **Tied.** Augmentation *content* is not what limits FN |
| `trainer_bndry.yaml` | Up-weighting the loss at cell-cell boundaries (α=5) should split merged cells | FN-recovery **71.1%** vs 71.9%; FP-suppression **identical** at 89.6%; AP 0.814 vs 0.820 (job 2702487) | **Tied.** The null is informative: 5× re-weighting caused *no* over-splitting, so the boundary evidence is not in the 8-bit pixels |
| `trainer_fixedscale.yaml` | Cellpose scales each image to its own [p1,p99], so the input scale depends on how many cells are in frame; a fixed scale should help sparse images | AP@0.5 **0.8023** vs 0.7996, **p=0.65**; dense +0.005, sparse −0.032 (job 3481718) | **Tied.** Note this arm was verified to actually apply — see the trap below |

Full statistics, job IDs and per-image breakdowns: `../notes/model_train_log.md`
(§2 for the June arms, §4d for fixed-scale).

## Running one of these again

They still work. The destination on the cluster is flat, so the command is
unchanged — `update_cluster.sh` copies `experiments/*.yaml` to `~/cellpose/`
alongside the active configs:

```bash
ssh pitt_crc 'cd ~/cellpose; \
  CELLPOSE_TRAIN_CONFIG=$HOME/cellpose/trainer_fnaug.yaml \
  FOLD_ROOT=$HOME/cellpose/runs/clc_cv_full3/clc_all RUN_TAG=<tag> \
    ./submit_cv.sh clc_all'
```

## The trap that makes `trainer_fixedscale.yaml` worth reading

Percentile normalisation is **affine-invariant**:

```
(x/s − p1/s) / (p99/s − p1/s)  ≡  (x − p1) / (p99 − p1)
```

so dividing an image by a constant and leaving cellpose's per-image
normalisation on changes *nothing* — the arm silently reproduces the baseline and
looks like a clean negative result. `trainer_slurm.py` therefore passes
`normalize=False` whenever `train.fixed_scale` is set, and the run log prints the
post-scaling input statistics so the arm can be verified rather than assumed.

The same applies at evaluation: `eval_seg.py --fixed_scale` mirrors it, and
`run_clc_cv.slurm` passes `EVAL_FIXED_SCALE` through automatically. **`clc_error_eval.py`
still hardcodes `normalize=True`** and cannot score a fixed-scale model correctly —
an error-eval submitted for this arm was cancelled for that reason, which is why
the arm has no FN-recovery figure.
