from typing import Tuple

from keras.models import Sequential
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense


def build_model(img_rows: int = 102, img_cols: int = 136, num_classes: int = 2) -> Tuple[Sequential, Tuple[int, int, int]]:
    input_shape = (img_rows, img_cols, 3)
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(6, (5, 5), activation="relu"),
        AveragePooling2D(pool_size=(2, 2), strides=2),
        Conv2D(16, (5, 5), activation="relu"),
        AveragePooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(120, activation="relu"),
        Dense(84, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model, input_shape


__all__ = ["build_model"]
