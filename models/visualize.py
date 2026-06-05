from typing import Optional
import os
import math

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image


def visualize_predictions(model_path: str, images_root: str, out_path: str = "predictions.png", img_rows: int = 102, img_cols: int = 136, num: int = 6):
    model = load_model(model_path)

    files = []
    for root, dirs, filenames in os.walk(images_root):
        for fn in sorted(filenames):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                fpath = os.path.join(root, fn)
                true_label = os.path.basename(root)
                files.append((fpath, true_label))
    if not files:
        raise ValueError(f"No images found under {images_root}")

    files = files[:num]
    imgs = []
    for fpath, _ in files:
        img = load_img(fpath, target_size=(img_rows, img_cols))
        imgs.append(img_to_array(img).astype("float32") / 255.0)

    batch = np.array(imgs)
    preds = model.predict(batch)
    pred_idxs = np.argmax(preds, axis=1)

    n = len(files)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)

    for i, ((fpath, true_label), pred_idx) in enumerate(zip(files, pred_idxs)):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        img = load_img(fpath, target_size=(img_rows, img_cols))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Pred: {pred_idx}\nTrue: {true_label}", fontsize=8)

    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


__all__ = ["visualize_predictions"]
