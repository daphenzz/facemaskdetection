import os
import tempfile

import numpy as np
from PIL import Image

from models.model import build_model
from models.prediction import predict_image


def make_image(path, size=(102, 136), color=(0, 255, 0)):
    img = Image.new("RGB", size, color)
    img.save(path)


def test_predict_image():
    # build and save a minimal model
    model, _ = build_model(img_rows=32, img_cols=32, num_classes=2)
    # create dummy weights by training on random data for 1 step
    X = np.random.rand(8, 32, 32, 3).astype("float32")
    y = np.zeros((8, 2))
    y[:4, 0] = 1
    y[4:, 1] = 1
    model.fit(X, y, epochs=1, batch_size=4, verbose=0)

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "m.h5")
        model.save(model_path)
        img_path = os.path.join(tmp, "test.jpg")
        make_image(img_path, size=(32, 32))
        pred = predict_image(model_path, img_path, img_rows=32, img_cols=32)
        assert isinstance(pred, int)
        assert pred in (0, 1)
