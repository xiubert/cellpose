import logging
import os
from datetime import datetime
from cellpose import io, models, train

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

model = models.CellposeModel(gpu=True)

weight_decay = 0.1
learning_rate = 1e-5
n_epochs = 95
model_name = "label_xfer_aug_retest"
batch_size = 8

# Refining an already-adapted model on a small target dataset
# weight_decay=0.05  # or 0.1 if overfitting appears
# learning_rate=1e-6
# n_epochs=30-50
# batch_size=8

logger.info(
    "train params: weight_decay=%s  learning_rate=%s  n_epochs=%s  "
    "model_name=%s  batch_size=%s",
    weight_decay, learning_rate, n_epochs, model_name, batch_size,
)


model_path, train_losses, test_losses = train.train_seg(model.net,
                            train_data=images, train_labels=labels,
                            test_data=test_images, test_labels=test_labels,
                            weight_decay=weight_decay, learning_rate=learning_rate,
                            n_epochs=n_epochs, model_name=model_name,
                            batch_size=batch_size)


# quick test
# model_path, train_losses, test_losses = train.train_seg(model.net,
#                             train_data=test_images, train_labels=test_labels,
#                             weight_decay=0.1, learning_rate=1e-5,
#                             n_epochs=80, model_name="label_xfer",
#                             batch_size=8)


# after training add model with: python -m cellpose --add_model "/path/to/model"