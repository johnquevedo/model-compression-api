"""Classification metrics: accuracy and macro-F1."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    preds_arr = np.asarray(preds)
    labels_arr = np.asarray(labels)
    return {
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "f1_macro": float(f1_score(labels_arr, preds_arr, average="macro")),
        "n_examples": int(labels_arr.shape[0]),
    }
