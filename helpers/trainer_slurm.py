import os
from cellpose import io, models, train
io.logger_setup()

train_dir = os.environ.get("CELLPOSE_TRAIN_DIR", os.path.expanduser("~/cellpose/train"))
test_dir = os.environ.get("CELLPOSE_TEST_DIR", os.path.expanduser("~/cellpose/test"))

output = io.load_train_test_data(train_dir, test_dir=test_dir, image_filter=None,
                                mask_filter="_seg.npy", look_one_level_down=False)
images, labels, image_names, test_images, test_labels, image_names_test = output

model = models.CellposeModel(gpu=True)

# A100 40 GB
model_path, train_losses, test_losses = train.train_seg(model.net,
                            train_data=images, train_labels=labels,
                            test_data=test_images, test_labels=test_labels,
                            weight_decay=0.1, learning_rate=1e-5,
                            n_epochs=85, model_name="label_xfer",
                            batch_size=8)

# quick test
# model_path, train_losses, test_losses = train.train_seg(model.net,
#                             train_data=test_images, train_labels=test_labels,
#                             weight_decay=0.1, learning_rate=1e-5,
#                             n_epochs=80, model_name="label_xfer",
#                             batch_size=8)


# after training add model with: python -m cellpose --add_model "/path/to/model"