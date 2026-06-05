import os
import tempfile

import numpy as np

from models.model import build_model
from models.evaluate import evaluate_model_on_data


def test_evaluate_creates_outputs():
    model, _ = build_model(img_rows=32, img_cols=32, num_classes=2)
    X = np.random.rand(8, 32, 32, 3).astype("float32")
    y = np.zeros((8, 2))
    y[:4, 0] = 1
    y[4:, 1] = 1
    model.fit(X, y, epochs=1, batch_size=4, verbose=0)

    with tempfile.TemporaryDirectory() as tmp:
        mpath = os.path.join(tmp, "m.h5")
        model.save(mpath)
        out = evaluate_model_on_data(mpath, X, y, classes=["with_mask", "without_mask"], out_prefix=os.path.join(tmp, "eval"))
        assert os.path.exists(out["confusion_matrix"])
        assert os.path.exists(out["report"])
        assert "precision" in out["report_text"]
