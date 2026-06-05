"""Preprocessing utilities for loading and preparing the face mask dataset.

Provide functions that load images from a directory and return numpy
arrays. This module intentionally does not run anything on import so it
can be reused by scripts and tests.
"""

import os
from typing import Tuple, List

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_images(data_path: str, img_rows: int = 102, img_cols: int = 136) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load images and integer labels from `data_path`.

    Args:
        data_path: Folder containing class subfolders (e.g. with_mask, without_mask).
        img_rows, img_cols: Target image size.

    Returns:
        (x, y, classes) where x is an array of images, y is integer labels, and
        classes is the ordered list of class names.
    """
    x, y = [], []
    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    if not classes:
        raise ValueError(f"No class subdirectories found in {data_path}")

    for label_idx, class_name in enumerate(classes):
        class_folder = os.path.join(data_path, class_name)
        for fname in sorted(os.listdir(class_folder)):
            fpath = os.path.join(class_folder, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                img = load_img(fpath, target_size=(img_rows, img_cols))
                arr = img_to_array(img)
                x.append(arr)
                y.append(label_idx)
            except Exception:
                # skip unreadable files
                continue

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return x, y, classes


def prepare_splits(x: np.ndarray, y: np.ndarray, num_classes: int = 2, test_size: float = 0.2,
                   val_size: float = 0.5, random_state: int = 42):
    """Split arrays into train/val/test and one-hot encode labels.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, X_val, X_test, y_train, y_val, y_test


__all__ = ["load_images", "prepare_splits"]
