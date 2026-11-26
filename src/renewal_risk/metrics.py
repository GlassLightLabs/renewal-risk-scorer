"""Model evaluation metrics, including the two a renewals leader actually asks about: lift and ARR capture."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def lift_at(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    """Churn rate in the riskiest `frac` of accounts divided by the overall churn rate."""
    n = max(1, int(round(len(p) * frac)))
    top = np.argsort(-p, kind="stable")[:n]
    base = y.mean()
    return float(y[top].mean() / base) if base > 0 else float("nan")


def arr_capture_at(y: np.ndarray, p: np.ndarray, arr: np.ndarray, frac: float) -> float:
    """Share of all churned ARR that sits in the riskiest `frac` of accounts."""
    n = max(1, int(round(len(p) * frac)))
    top = np.argsort(-p, kind="stable")[:n]
    lost = (arr * y).sum()
    return float((arr[top] * y[top]).sum() / lost) if lost > 0 else float("nan")


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            rows.append(
                {
                    "predicted": round(float(p[chunk].mean()), 4),
                    "actual": round(float(y[chunk].mean()), 4),
                    "n": int(len(chunk)),
                }
            )
    return rows


def classification_report(y: np.ndarray, p: np.ndarray, arr: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    arr = np.asarray(arr, dtype=float)
    if y.min() == y.max():
        return {"n": int(len(y)), "positives": int(y.sum()), "note": "single class; metrics undefined"}
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "lift_top_10pct": round(lift_at(y, p, 0.10), 2),
        "arr_capture_top_20pct": round(arr_capture_at(y, p, arr, 0.20), 4),
        "calibration": calibration_table(y, p),
    }
