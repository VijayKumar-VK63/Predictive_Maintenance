"""Evaluation module for model assessment."""

from pathlib import Path
from typing import Dict, Optional
import json
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
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Dictionary of metrics.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Confusion matrix as 2x2 array.
    """
    return confusion_matrix(y_true, y_pred)


def threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Analyze metrics across different probability thresholds.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for positive class.
        thresholds: Array of thresholds to evaluate. Defaults to 0.01 to 0.99.

    Returns:
        DataFrame with metrics at each threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    rows = []
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        n_predicted_failures = int(y_pred_thresh.sum())

        rows.append(
            {
                "threshold": thresh,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_failures": n_predicted_failures,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }
        )

    return pd.DataFrame(rows)


def find_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 score.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.

    Returns:
        Optimal threshold.
    """
    df = threshold_analysis(y_true, y_prob)
    best_idx = df["f1"].idxmax()
    return float(df.loc[best_idx, "threshold"])


def find_threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.8) -> float:
    """Find lowest threshold achieving at least target recall.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        target_recall: Minimum recall to achieve.

    Returns:
        Threshold achieving target recall, or 0.5 if not possible.
    """
    df = threshold_analysis(y_true, y_prob)
    candidates = df[df["recall"] >= target_recall]
    if candidates.empty:
        return 0.5
    return float(candidates.loc[candidates["precision"].idxmax(), "threshold"])


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Failure"],
        yticklabels=["Normal", "Failure"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot ROC curve.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot Precision-Recall curve.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(y=y_true.mean(), color="k", linestyle="--", label=f"Baseline (failure rate = {y_true.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_threshold_analysis(df_thresh: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot metrics vs threshold.

    Args:
        df_thresh: DataFrame from threshold_analysis.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(df_thresh["threshold"], df_thresh["precision"], label="Precision")
    axes[0, 0].plot(df_thresh["threshold"], df_thresh["recall"], label="Recall")
    axes[0, 0].plot(df_thresh["threshold"], df_thresh["f1"], label="F1")
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Metrics vs Threshold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df_thresh["threshold"], df_thresh["predicted_failures"])
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Predicted Failures vs Threshold")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df_thresh["threshold"], df_thresh["true_positives"], label="TP")
    axes[1, 0].plot(df_thresh["threshold"], df_thresh["false_positives"], label="FP")
    axes[1, 0].plot(df_thresh["threshold"], df_thresh["false_negatives"], label="FN")
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Confusion Matrix Components vs Threshold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df_thresh["recall"], df_thresh["precision"])
    axes[1, 1].set_xlabel("Recall")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].set_title("Precision-Recall Tradeoff")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics.
        path: Output path.
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_model_comparison(comparison_df: pd.DataFrame, path: Path) -> None:
    """Save model comparison table to CSV.

    Args:
        comparison_df: DataFrame with model comparison.
        path: Output path.
    """
    comparison_df.to_csv(path, index=False)