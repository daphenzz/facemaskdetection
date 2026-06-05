from typing import Sequence
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray, classes: Sequence[str], out_path: str):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def evaluate_model_on_data(model_path: str, X: np.ndarray, y: np.ndarray, classes: Sequence[str], out_prefix: str = "eval"):
    model = load_model(model_path)
    Xp = X.astype("float32")
    if Xp.max() > 2.0:
        Xp = Xp / 255.0

    preds = model.predict(Xp)
    pred_idx = np.argmax(preds, axis=1)
    true_idx = np.argmax(y, axis=1)

    cm = confusion_matrix(true_idx, pred_idx)
    cm_path = f"{out_prefix}_confusion.png"
    plot_confusion_matrix(cm, classes, cm_path)

    report = classification_report(true_idx, pred_idx, target_names=list(classes))
    txt_path = f"{out_prefix}_report.txt"
    with open(txt_path, "w") as f:
        f.write(report)

    return {"confusion_matrix": cm_path, "report": txt_path, "report_text": report}


__all__ = ["evaluate_model_on_data", "plot_confusion_matrix"]
