import argparse
import os
from typing import List

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img


def predict_image(model_path: str, image_path: str, img_rows: int = 102, img_cols: int = 136) -> int:
    model = load_model(model_path)
    img = load_img(image_path, target_size=(img_rows, img_cols))
    arr = img_to_array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    return int(np.argmax(preds, axis=1)[0])


def batch_predict(model_path: str, image_paths: List[str], img_rows: int = 102, img_cols: int = 136) -> List[int]:
    model = load_model(model_path)
    batch = []
    for p in image_paths:
        img = load_img(p, target_size=(img_rows, img_cols))
        batch.append(img_to_array(img))
    batch = np.array(batch, dtype="float32")
    preds = model.predict(batch)
    return [int(x) for x in np.argmax(preds, axis=1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved Keras model (.h5)")
    parser.add_argument("--image", help="Path to a single image to predict")
    parser.add_argument("--dir", help="Directory of images to run batch prediction on")
    args = parser.parse_args()

    if args.image:
        idx = predict_image(args.model, args.image)
        print(idx)
    elif args.dir:
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if os.path.isfile(os.path.join(args.dir, f))]
        preds = batch_predict(args.model, sorted(files))
        for p, lab in zip(sorted(files), preds):
            print(p, lab)
    else:
        parser.error("Either --image or --dir must be provided.")


if __name__ == "__main__":
    main()
