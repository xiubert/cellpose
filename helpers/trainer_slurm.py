import logging
import os
import sys
from datetime import datetime

import yaml
from cellpose import io, models, train

# helpers/ on path so aug_online imports regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aug_online

slurm_job_id = os.environ.get("SLURM_JOB_ID")
logfile_name = f"run.{slurm_job_id}.log" if slurm_job_id else f"run.{datetime.now().strftime('%Y%m%d_%H%M')}.log"
io.logger_setup(logfile_name=logfile_name)

# REQUIRES: 1 GPU with at least 40 GB VRAM, e.g. NVIDIA A100 40 GB

logger = logging.getLogger(__name__)

train_dir = os.environ.get("CELLPOSE_TRAIN_DIR", os.path.expanduser("~/cellpose/train"))
test_dir = os.environ.get("CELLPOSE_TEST_DIR", os.path.expanduser("~/cellpose/test"))
logger.info("train_dir: %s", train_dir)
logger.info("test_dir:  %s", test_dir)

output = io.load_train_test_data(train_dir, test_dir=test_dir, image_filter=None,
                                mask_filter="_seg.npy", look_one_level_down=False)
images, labels, image_names, test_images, test_labels, image_names_test = output

logger.info("train images (%d): %s", len(image_names), image_names)
logger.info("test  images (%d): %s", len(image_names_test or []), image_names_test)

# Warm-start: init from an existing model instead of stock cpsam.
# From env CELLPOSE_PRETRAINED (a model path) — used by the CV loop for the
# "refine label_xfer_aug_retest on CLC" recipe. Default: stock cpsam.
pretrained = os.environ.get("CELLPOSE_PRETRAINED")
if pretrained:
    logger.info("warm-start: init from pretrained_model=%s", pretrained)
    model = models.CellposeModel(gpu=True, pretrained_model=pretrained)
else:
    model = models.CellposeModel(gpu=True)

# --- Hyperparameters from YAML (single source of truth; see trainer.yaml) ---
DEFAULTS = {
    "weight_decay": 0.1,
    "learning_rate": 1e-5,
    "n_epochs": 95,
    "batch_size": 8,
    "model_name": "label_xfer_aug_retest",
    "nimg_per_epoch": None,   # crops sampled per epoch; None -> #train files
}

config_path = os.environ.get(
    "CELLPOSE_TRAIN_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer.yaml"),
)

cfg = dict(DEFAULTS)
augment_cfg = {}
if os.path.exists(config_path):
    with open(config_path) as f:
        full = yaml.safe_load(f) or {}
    loaded = full.get("train") or {}
    augment_cfg = full.get("augment") or {}
    unknown = set(loaded) - set(DEFAULTS)
    if unknown:
        logger.warning("ignoring unknown train config keys: %s", sorted(unknown))
    cfg.update({k: v for k, v in loaded.items() if k in DEFAULTS})
    logger.info("loaded train config: %s", config_path)
else:
    logger.warning("config %s not found — using built-in defaults", config_path)

# Online image-only augmentation (None if no augment: block / nothing enabled)
img_transform = aug_online.make(augment_cfg)
logger.info("augment config: %s", augment_cfg or "(none — img_transform disabled)")

weight_decay = cfg["weight_decay"]
learning_rate = cfg["learning_rate"]
n_epochs = cfg["n_epochs"]
# CELLPOSE_MODEL_NAME overrides the YAML model_name (CV loop sets it per fold).
model_name = os.environ.get("CELLPOSE_MODEL_NAME") or cfg["model_name"]
batch_size = cfg["batch_size"]
nimg_per_epoch = cfg["nimg_per_epoch"]

logger.info(
    "train params: weight_decay=%s  learning_rate=%s  n_epochs=%s  "
    "model_name=%s  batch_size=%s  nimg_per_epoch=%s",
    weight_decay, learning_rate, n_epochs, model_name, batch_size, nimg_per_epoch,
)


model_path, train_losses, test_losses = train.train_seg(model.net,
                            train_data=images, train_labels=labels,
                            test_data=test_images, test_labels=test_labels,
                            weight_decay=weight_decay, learning_rate=learning_rate,
                            n_epochs=n_epochs, model_name=model_name,
                            nimg_per_epoch=nimg_per_epoch,
                            batch_size=batch_size, img_transform=img_transform)


# quick test
# model_path, train_losses, test_losses = train.train_seg(model.net,
#                             train_data=test_images, train_labels=test_labels,
#                             weight_decay=0.1, learning_rate=1e-5,
#                             n_epochs=80, model_name="label_xfer",
#                             batch_size=8)


# after training add model with: python -m cellpose --add_model "/path/to/model"