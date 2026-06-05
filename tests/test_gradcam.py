import os
import tempfile

import numpy as np
from PIL import Image

from models.model import build_model
from models.gradcam import grad_cam


def make_image(path, size=(32, 32), color=(200, 100, 50)):
    img = Image.new("RGB", size, color)
    img.save(path)


def test_gradcam_generates_image():
    model, _ = build_model(img_rows=32, img_cols=32, num_classes=2)
    # quick fake training
    X = np.random.rand(8, 32, 32, 3).astype("float32")
    y = np.zeros((8, 2))
    y[:4, 0] = 1
    y[4:, 1] = 1
    model.fit(X, y, epochs=1, batch_size=4, verbose=0)

    with tempfile.TemporaryDirectory() as tmp:
        mpath = os.path.join(tmp, "m.h5")
        model.save(mpath)
        img_path = os.path.join(tmp, "in.jpg")
        make_image(img_path, size=(32, 32))
        out = os.path.join(tmp, "gc.png")
        res = grad_cam(mpath, img_path, out_path=out, img_rows=32, img_cols=32)
        assert os.path.exists(res)
        assert os.path.getsize(res) > 0
