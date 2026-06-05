"""Utility helpers for project-wide utilities like reproducible seeds."""
import os
import random

import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for python, numpy, and environment to improve reproducibility.

    Note: full determinism with TensorFlow may require additional environment
    settings; this provides a reasonable, portable starting point for tests.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


__all__ = ["set_seed"]
