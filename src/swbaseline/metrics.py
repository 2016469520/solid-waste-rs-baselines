from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    total = tp + tn + fp + fn
    po = accuracy
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total) if total else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "kappa": kappa,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[: len(pos)]
    return float((np.sum(pos_ranks) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    positives = np.sum(y_sorted == 1)
    if positives == 0:
        return 0.0
    tp = np.cumsum(y_sorted == 1)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float(np.sum(precision[y_sorted == 1]) / positives)


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Iterable[float]) -> List[Dict[str, float]]:
    rows = []
    for threshold in thresholds:
        rows.append(binary_metrics(y_true, y_prob, float(threshold)))
    return rows


def best_by(rows: List[Dict[str, float]], key: str = "f1") -> Dict[str, float]:
    return max(rows, key=lambda row: row.get(key, 0.0))

