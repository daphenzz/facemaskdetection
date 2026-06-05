import os
from typing import Optional

try:
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
    from keras.preprocessing.image import ImageDataGenerator

from data.dataset import get_splits
from .model import build_model
from utils import set_seed
import numpy as np


def train(data_path: str, out: str = "models/model.h5", epochs: int = 10, batch_size: int = 32, seed: int = 42, use_generator: bool = False, logdir: Optional[str] = None):
    set_seed(seed)
    X_train, X_val, X_test, y_train, y_val, y_test, classes = get_splits(data_path)
    model, _ = build_model()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    callbacks = [ModelCheckpoint(out, monitor="val_loss", save_best_only=True), EarlyStopping(monitor="val_loss", patience=3)]
    if logdir:
        callbacks.append(TensorBoard(log_dir=logdir))

    if use_generator:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                           rotation_range=15,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.1,
                                           zoom_range=0.1,
                                           horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_gen = val_datagen.flow(X_val, y_val, batch_size=batch_size)

        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    else:
        history = model.fit(X_train / 255.0, y_train, batch_size=batch_size, validation_data=(X_val / 255.0, y_val), epochs=epochs, callbacks=callbacks)

    score = model.evaluate(X_test / 255.0, y_test, verbose=0)
    return {"loss": float(score[0]), "accuracy": float(score[1])}


__all__ = ["train"]
