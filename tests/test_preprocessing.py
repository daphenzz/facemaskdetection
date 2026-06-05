import os
import tempfile

import numpy as np
from PIL import Image

from data.preprocessing import load_images, prepare_splits


def make_image(path, size=(102, 136), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    img.save(path)


def test_load_and_split():
    with tempfile.TemporaryDirectory() as tmp:
        class_dirs = [os.path.join(tmp, "with_mask"), os.path.join(tmp, "without_mask")]
        for d in class_dirs:
            os.makedirs(d, exist_ok=True)
        # create 4 images per class
        for i in range(4):
            make_image(os.path.join(class_dirs[0], f"img_{i}.jpg"))
            make_image(os.path.join(class_dirs[1], f"img_{i}.jpg"))

        x, y, classes = load_images(tmp, img_rows=32, img_cols=32)
        assert x.shape[0] == 8
        assert set(classes) == {"with_mask", "without_mask"}

        X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(x, y, num_classes=2, test_size=0.5, val_size=0.5)
        # with these params, train should be half (4) then val/test 2 each
        assert X_train.shape[0] == 4
        assert X_val.shape[0] == 2
        assert X_test.shape[0] == 2
