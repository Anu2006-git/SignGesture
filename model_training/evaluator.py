"""
model_training/evaluator.py
===========================
Model evaluation, performance metrics computation, confusion matrix
plotting, and report generation for Sign Language classification.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/saving
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import ALPHABETS, EVALUATION_DIR, NUM_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Stores full evaluation metrics for a model."""
    model_name: str
    total_test_samples: int
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    confusion_matrix: np.ndarray
    classification_report_text: str
    per_class_f1: Dict[str, float] = field(default_factory=dict)


class ModelEvaluator:
    """Computes comprehensive evaluation metrics on unseen test sets."""

    def __init__(self, output_dir: Path = EVALUATION_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        active_classes: Optional[List[int]] = None,
    ) -> ModelMetrics:
        """
        Calculates all standard evaluation metrics and generates
        a confusion matrix heatmap on the unseen test set.
        """
        if len(y_true) == 0:
            raise ValueError(f"Cannot evaluate {model_name}: test set is empty.")

        # Accuracy
        acc = float(accuracy_score(y_true, y_pred))

        # Precision, Recall, F1 (both macro and weighted)
        p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))

        r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

        f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Full 26x26 confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

        # Determine populated labels present in test set
        unique_labels = sorted(list(set(y_true).union(set(y_pred))))
        target_names = [ALPHABETS[i] for i in unique_labels]

        report_txt = classification_report(
            y_true,
            y_pred,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0,
        )

        # Per-class F1
        per_class_scores = f1_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
        per_class_f1 = {ALPHABETS[lbl]: float(score) for lbl, score in zip(unique_labels, per_class_scores)}

        # Plot and save confusion matrix
        self._plot_confusion_matrix(cm, model_name, unique_labels)

        return ModelMetrics(
            model_name=model_name,
            total_test_samples=len(y_true),
            accuracy=acc,
            precision_macro=p_macro,
            precision_weighted=p_weighted,
            recall_macro=r_macro,
            recall_weighted=r_weighted,
            f1_macro=f1_mac,
            f1_weighted=f1_wt,
            confusion_matrix=cm,
            classification_report_text=report_txt,
            per_class_f1=per_class_f1,
        )

    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str, active_labels: List[int]) -> Path:
        """Plots and saves a high-resolution Confusion Matrix heatmap."""
        plt.figure(figsize=(10, 8))

        # Focus plot on active labels or all 26
        if len(active_labels) > 0 and len(active_labels) < 26:
            subset_cm = cm[np.ix_(active_labels, active_labels)]
            tick_marks = np.arange(len(active_labels))
            tick_labels = [ALPHABETS[i] for i in active_labels]
        else:
            subset_cm = cm
            tick_marks = np.arange(NUM_CLASSES)
            tick_labels = ALPHABETS

        plt.imshow(subset_cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix — {model_name} (Unseen Test Set)", fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.xticks(tick_marks, tick_labels, rotation=0, fontsize=10)
        plt.yticks(tick_marks, tick_labels, fontsize=10)
        plt.xlabel("Predicted Alphabet", fontsize=11, labelpad=8)
        plt.ylabel("True Alphabet", fontsize=11, labelpad=8)

        # Annotate cell counts
        thresh = subset_cm.max() / 2.0 if subset_cm.max() > 0 else 1.0
        for i in range(subset_cm.shape[0]):
            for j in range(subset_cm.shape[1]):
                val = subset_cm[i, j]
                plt.text(
                    j, i, f"{val}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if val > thresh else "black",
                    fontsize=9,
                )

        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        save_path = self.output_dir / f"confusion_matrix_{safe_name}.png"
        plt.savefig(str(save_path), dpi=200)
        plt.close()
        return save_path

    def plot_cnn_learning_curves(self, history: Dict[str, List[float]]) -> Optional[Path]:
        """Plots CNN loss and accuracy curves over epochs."""
        if not history.get("train_loss"):
            return None

        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(12, 5))

        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=1.5, markersize=4)
        if history.get("val_loss") and len(history["val_loss"]) == len(epochs):
            plt.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=1.5, markersize=4)
        plt.title("Model Loss Curve", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        # Accuracy curve
        plt.subplot(1, 2, 2)
        if history.get("train_acc") and len(history["train_acc"]) == len(epochs):
            plt.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc", linewidth=1.5, markersize=4)
        if history.get("val_acc") and len(history["val_acc"]) == len(epochs):
            plt.plot(epochs, [a * 100 for a in history["val_acc"]], "g--s", label="Val Acc", linewidth=1.5, markersize=4)
        plt.title("Model Accuracy Curve", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        plt.tight_layout()
        save_path = self.output_dir / "cnn_learning_curves.png"
        plt.savefig(str(save_path), dpi=200)
        plt.close()
        return save_path
