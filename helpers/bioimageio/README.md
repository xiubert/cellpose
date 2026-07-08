# CLC mouse cochlear hair-cell segmentation (Cellpose-SAM)

Cellpose-SAM model fine-tuned from stock `cpsam` to segment mouse cochlear hair
cells using MYO7A staining (Chris Cunningham / CLC / dataset including adult + neonate,
8/16/32 kHz tonotopic regions). Trained on all 67 curated images (20 animals).

**Leave-one-animal-out CV performance (held-out, full set):**
- AP@0.5 ~ 0.82
- False-negative recovery ~ 72%, false-positive suppression ~ 90% vs the
  model-assisted ground truth.

**Output:** standard Cellpose flow representation - 3 channels [dY, dX,
cellprob]. Instance masks are produced by cellpose's flow-following dynamics
(not part of the network forward).

**Attribution:** data and labeling - Chris Cunningham Lab, University of
Pittsburgh; model training - Patrick Cody, University of Pittsburgh. Built on
Cellpose-SAM. Fork: https://github.com/xiubert/cellpose/tree/label_xfer_ihc_ohc

See helpers/notes/clc_seg_results.md and progress.md for the full derivation.
