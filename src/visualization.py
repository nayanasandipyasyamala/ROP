"""
Stage 10: Publication-Ready Visualization and Plotting Module for ROP Experiments.

Generates:
1. Loss and Accuracy Learning Curves.
2. Confusion Matrices (normalized and count-based).
3. ROC Curves and Precision-Recall Curves with AUROC/AUPRC labels.
4. Cross-Dataset Performance Comparison Bar Charts (A1 vs A2 vs A3 vs A4 vs A5).
5. Imbalance-Handling Method Comparison Charts (Cross-Entropy vs Weighted CE vs Focal Loss vs Balanced Sampling).
6. Backbone Architecture Comparison Charts (ResNet-50 vs EfficientNet-B0 vs ConvNeXt-Tiny).
7. Preprocessing comparison images (raw vs cropped & padded).
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config


def plot_learning_curves(history: Dict[str, List[float]], save_path: Path, title: str = "Training Curves"):
    """Plot Loss and Accuracy curves across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=1.5, markersize=4)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=1.5, markersize=4)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.set_title(f"{title} - Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True)

    # Accuracy / AUROC
    ax2.plot(epochs, history["val_accuracy"], "g-^", label="Val Accuracy", linewidth=1.5, markersize=4)
    if "val_auroc" in history and any(x > 0 for x in history["val_auroc"]):
        ax2.plot(epochs, history["val_auroc"], "m--d", label="Val AUROC", linewidth=1.5, markersize=4)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric", fontsize=11, fontweight="bold")
    ax2.set_title(f"{title} - Metrics", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = False
):
    """Plot formatted confusion matrix with cell annotations."""
    cm_arr = np.array(cm)
    if normalize:
        cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]
        disp_arr = cm_norm
        fmt = ".2%"
    else:
        disp_arr = cm_arr
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.imshow(disp_arr, cmap="Blues", interpolation="nearest")
    plt.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")

    thresh = disp_arr.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = disp_arr[i, j]
            txt = f"{val:{fmt}}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    title: str = "ROC & Precision-Recall Curves"
):
    """Plot both ROC and PR curves side-by-side."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # ROC
    ax1.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC curve (AUC = {auroc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10, fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True)

    # PR
    baseline = np.mean(y_true)
    ax2.plot(recall, precision, color="#d95f02", lw=2, label=f"PR curve (AUPRC = {auprc:.3f})")
    ax2.plot([0, 1], [baseline, baseline], color="gray", lw=1, linestyle="--", label=f"Prevalence ({baseline:.2f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax2.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


def plot_preprocessing_comparison(raw_path: str, preprocessed_path: str, save_path: Path):
    """Plot side-by-side comparison of raw fundus vs preprocessed (cropped + padded)."""
    raw_img = Image.open(raw_path).convert("RGB")
    prep_img = Image.open(preprocessed_path).convert("RGB")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    ax1.imshow(raw_img)
    ax1.set_title(f"Raw Input ({raw_img.size[0]}x{raw_img.size[1]})", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(prep_img)
    ax2.set_title(f"Preprocessed ({prep_img.size[0]}x{prep_img.size[1]})", fontsize=11, fontweight="bold")
    ax2.axis("off")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")
