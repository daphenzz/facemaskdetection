#!/usr/bin/env python3
"""Generate three montages:
 - maskesiz_dogru.png: images without mask correctly predicted
 - maskeli_yanlis.png: images with mask but predicted as without_mask (wrong)
 - maskesiz_yanlis.png: images without mask but predicted as with_mask (wrong)

Saves outputs into the latest `results/` subfolder if present, otherwise `outputs/`.
"""
import os
import math
from pathlib import Path
from typing import List

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


def find_latest_results_dir(base="results"):
    if not os.path.isdir(base):
        return "outputs"
    subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not subs:
        return "outputs"
    subs = sorted(subs)
    return os.path.join(base, subs[-1])


def collect_files(images_root: str):
    files = []
    for root, _, filenames in os.walk(images_root):
        for fn in sorted(filenames):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                files.append((os.path.join(root, fn), os.path.basename(root)))
    return files


def make_montage(paths: List[str], out_path: str, img_rows=102, img_cols=136, cols=4):
    if not paths:
        return None
    n = len(paths)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for i, p in enumerate(paths):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        img = load_img(p, target_size=(img_rows, img_cols))
        ax.imshow(img)
        ax.axis("off")
    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        axes[r, c].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.h5")
    parser.add_argument("--data", default="data/images")
    parser.add_argument("--out_dir", default=None, help="Directory to save montages (defaults to latest results/ or outputs/)")
    args = parser.parse_args()

    out_dir = args.out_dir or find_latest_results_dir()
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(args.model)
    files = collect_files(args.data)
    if not files:
        raise SystemExit(f"No images found under {args.data}")

    # determine classes order by reading data folder names sorted
    classes = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    idx_with = classes.index("with_mask") if "with_mask" in classes else 0
    idx_without = classes.index("without_mask") if "without_mask" in classes else 1

    paths_maskesiz_dogru = []
    paths_maskeli_yanlis = []
    paths_maskesiz_yanlis = []

    batch_imgs = []
    batch_meta = []
    for fpath, true_label in files:
        img = img_to_array(load_img(fpath, target_size=(102, 136))).astype("float32") / 255.0
        batch_imgs.append(img)
        batch_meta.append((fpath, true_label))

    batch = np.array(batch_imgs)
    preds = model.predict(batch, verbose=0)
    pred_idxs = np.argmax(preds, axis=1)

    for (fpath, true_label), pred in zip(batch_meta, pred_idxs):
        if true_label == "without_mask":
            if pred == idx_without:
                paths_maskesiz_dogru.append(fpath)
            else:
                paths_maskesiz_yanlis.append(fpath)
        elif true_label == "with_mask":
            if pred == idx_without:
                paths_maskeli_yanlis.append(fpath)

    # pick up to 12 images each for readability
    paths_maskesiz_dogru = paths_maskesiz_dogru[:12]
    paths_maskeli_yanlis = paths_maskeli_yanlis[:12]
    paths_maskesiz_yanlis = paths_maskesiz_yanlis[:12]

    out1 = os.path.join(out_dir, "maskesiz_dogru.png")
    out2 = os.path.join(out_dir, "maskeli_yanlis.png")
    out3 = os.path.join(out_dir, "maskesiz_yanlis.png")

    r1 = make_montage(paths_maskesiz_dogru, out1)
    r2 = make_montage(paths_maskeli_yanlis, out2)
    r3 = make_montage(paths_maskesiz_yanlis, out3)

    print("Saved:")
    if r1:
        print(r1)
    else:
        print("No maskesiz_dogru images found")
    if r2:
        print(r2)
    else:
        print("No maskeli_yanlis images found")
    if r3:
        print(r3)
    else:
        print("No maskesiz_yanlis images found")


if __name__ == "__main__":
    main()
