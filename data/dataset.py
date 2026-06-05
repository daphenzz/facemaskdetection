"""Dataset helpers for facemask detection project.

This module provides a thin wrapper around `data.preprocessing` to return
train/val/test splits ready for model consumption.
"""
from typing import Tuple

import numpy as np

from .preprocessing import load_images, prepare_splits


def get_splits(data_path: str, img_rows: int = 102, img_cols: int = 136, num_classes: int = 2,
               test_size: float = 0.2, val_size: float = 0.5, random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """Load images and return (X_train, X_val, X_test, y_train, y_val, y_test, classes)

    Args mirror those in `load_images` and `prepare_splits`.
    """
    x, y, classes = load_images(data_path, img_rows=img_rows, img_cols=img_cols)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(x, y, num_classes=num_classes,
                                                                    test_size=test_size, val_size=val_size,
                                                                    random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test, classes
