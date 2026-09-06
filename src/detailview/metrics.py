"""Metric helpers for DetailView benchmark."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def overall_metrics(y_true, y_pred, labels: list[int]) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return {
            "N": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
        }
    acc = float(accuracy_score(y_true, y_pred))
    try:
        bacc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bacc = float("nan")
    return {
        "N": int(len(y_true)),
        "accuracy": 100.0 * acc,
        "balanced_accuracy": 100.0 * bacc,
        "macro_precision": 100.0
        * float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": 100.0
        * float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": 100.0
        * float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": 100.0
        * float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def per_class_rows(
    y_true,
    y_pred,
    labels: list[int],
    id_to_name: dict,
    *,
    site: str,
    method: str,
    train_n: dict[int, int],
) -> list[dict]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return []
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    rows = []
    for i, lab in enumerate(labels):
        rows.append(
            {
                "site": site,
                "method": method,
                "species_id": int(lab),
                "species": id_to_name.get(int(lab), str(lab)),
                "support": int(support[i]),
                "precision": 100.0 * float(p[i]),
                "recall": 100.0 * float(r[i]),
                "f1": 100.0 * float(f1[i]),
                "forspecies_train_n": int(train_n.get(int(lab), -1)),
            }
        )
    return rows


def confusion_counts(y_true, y_pred, labels: list[int]) -> np.ndarray:
    return confusion_matrix(
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        labels=labels,
    )


def bootstrap_oa_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(float(np.mean(y_true[idx] == y_pred[idx])))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return 100.0 * float(lo), 100.0 * float(hi)


def height_bin_label(h: float, width: float = 2.0) -> str:
    if not np.isfinite(h):
        return "NA"
    lo = int(np.floor(h / width) * width)
    hi = lo + int(width)
    return f"({lo}, {hi}]" if h > lo else f"[{lo}, {hi}]"
