"""Test-split metrics and plots shared by every model, so all results follow the same definitions."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data import LABEL_NAMES

# Prediction recorded when an answer names none of the four topics. It counts as wrong.
INVALID = "INVALID"


def summarize(predictions: pd.DataFrame) -> dict:
    """Return the comparison metrics for one model's per-article predictions.

    predictions needs the columns true_label, predicted_label (a topic name, or INVALID), latency_s, and
    cost_usd. Macro-F1 averages the four topics; an INVALID prediction counts as a miss for its true topic.
    """
    y_true, y_pred = predictions.true_label, predictions.predicted_label
    return {
        "n": len(predictions),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABEL_NAMES, average="macro", zero_division=0),
        "invalid": int((y_pred == INVALID).sum()),
        "cost_per_1000_usd": predictions.cost_usd.mean() * 1000,
        "latency_mean_s": predictions.latency_s.mean(),
        "latency_median_s": predictions.latency_s.median(),
    }


def save_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), LABEL_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(4), LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=10)
    for i, j in itertools.product(range(4), range(4)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
