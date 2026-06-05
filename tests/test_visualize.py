import os
import tempfile

import numpy as np
from PIL import Image

from models.model import build_model
from models.visualize import visualize_predictions


def make_image(path, size=(32, 32), color=(0, 128, 255)):
    img = Image.new("RGB", size, color)
    img.save(path)


def test_visualize_predictions_creates_file():
    # build and save a minimal model
    model, _ = build_model(img_rows=32, img_cols=32, num_classes=2)
    X = np.random.rand(8, 32, 32, 3).astype("float32")
    y = np.zeros((8, 2))
    y[:4, 0] = 1
    y[4:, 1] = 1
    model.fit(X, y, epochs=1, batch_size=4, verbose=0)

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "m.h5")
        model.save(model_path)

        # create simple image tree
        class_dirs = [os.path.join(tmp, "with_mask"), os.path.join(tmp, "without_mask")]
        for d in class_dirs:
            os.makedirs(d, exist_ok=True)
        for i in range(3):
            make_image(os.path.join(class_dirs[0], f"img_{i}.jpg"))
            make_image(os.path.join(class_dirs[1], f"img_{i}.jpg"))

        out = os.path.join(tmp, "out.png")
        res = visualize_predictions(model_path, tmp, out_path=out, img_rows=32, img_cols=32, num=6)
        assert os.path.exists(res)
        assert os.path.getsize(res) > 0
