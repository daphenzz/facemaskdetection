"""Gradient-based saliency visualization for models in the top-level package."""
from typing import Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img


def _prepare_heatmap(gradients: np.ndarray) -> np.ndarray:
    heatmap = np.abs(gradients).mean(axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return np.uint8(255 * heatmap)


def grad_cam(
    model_path: str,
    image_path: str,
    out_path: str = "gradcam.png",
    img_rows: int = 102,
    img_cols: int = 136,
    class_index: Optional[int] = None,
):
    model = load_model(model_path)

    img = load_img(image_path, target_size=(img_rows, img_cols))
    x = img_to_array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    x_tf = tf.convert_to_tensor(x)

    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        predictions = model(x_tf, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]).numpy())
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, x_tf)
    if grads is None:
        raise ValueError("Could not compute gradient heatmap for the provided model and image.")

    heatmap = _prepare_heatmap(grads[0].numpy())
    heatmap_img = Image.fromarray(heatmap).resize((img_cols, img_rows)).convert("L")

    base_img = Image.open(image_path).resize((img_cols, img_rows)).convert("RGBA")
    cmap = plt.get_cmap("jet")
    colored = cmap(np.array(heatmap_img) / 255.0)
    colored_img = Image.fromarray((colored[:, :, :3] * 255).astype("uint8")).convert("RGBA")
    overlay = Image.blend(base_img, colored_img, alpha=0.5)
    overlay.save(out_path)
    return out_path


__all__ = ["grad_cam"]
