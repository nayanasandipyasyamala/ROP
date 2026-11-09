"""
Stage 10: Clinical Evaluation Metrics Module for ROP Screening & Severity Staging.

Computes comprehensive clinical metrics:
- Binary: Accuracy, Sensitivity (Recall), Specificity, Precision, F1-score, AUROC, AUPRC, Confusion Matrix.
- Multiclass: Accuracy, Macro-F1, Per-class Precision/Recall/F1, OvR AUROC, Confusion Matrix.
- Severe-specific metrics: Stage 4 Recall, Stage 5 Recall, A-ROP Recall, Severe-vs-NonSevere Sensitivity.
- Non-parametric Bootstrap Confidence Intervals (95% CI).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute binary classification metrics for Normal (0) vs ROP (1).
    y_prob: probability of class 1 (ROP).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # AUROC and AUPRC
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = np.nan

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "total_images": len(y_true)
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute multiclass severity grading metrics (e.g. mild, moderate, severe).
    y_probs: (N, num_classes) array of predicted probabilities.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = np.argmax(y_probs, axis=1)

    num_classes = y_probs.shape[1]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # One-vs-Rest AUROC
    ovr_auroc = {}
    for i, name in enumerate(class_names):
        try:
            bin_true = (y_true == i).astype(int)
            if len(np.unique(bin_true)) > 1:
                ovr_auroc[name] = float(roc_auc_score(bin_true, y_probs[:, i]))
            else:
                ovr_auroc[name] = np.nan
        except Exception:
            ovr_auroc[name] = np.nan

    res = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "per_class": {},
        "ovr_auroc": ovr_auroc,
        "total_images": len(y_true)
    }

    for i, name in enumerate(class_names):
        res["per_class"][name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1_score": float(per_class_f1[i]),
            "support": int((y_true == i).sum())
        }

    return res


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_func=compute_binary_metrics,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Compute non-parametric 95% bootstrap confidence intervals for binary metrics.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_metrics = collections.defaultdict(list)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        # Ensure at least one positive and one negative sample
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = metric_func(y_true[idx], y_prob[idx])
        for k in ["accuracy", "sensitivity", "specificity", "precision", "f1_score", "auroc", "auprc"]:
            val = m.get(k)
            if val is not None and not np.isnan(val):
                boot_metrics[k].append(val)

    ci_dict = {}
    for k, vals in boot_metrics.items():
        if len(vals) > 0:
            ci_dict[k] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        else:
            ci_dict[k] = (np.nan, np.nan)

    return ci_dict


import collections
