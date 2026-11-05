"""
model_training/comparison.py
============================
Model comparison engine that contrasts image-based CNN and
landmark-based Random Forest on identical unseen test sets.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .evaluator import ModelMetrics


@dataclass
class ComparisonResult:
    """Outcome of model comparison."""
    winning_model_name: str
    winning_model_type: str    # "CNN" or "RandomForest"
    margin_accuracy: float
    margin_f1: float
    selection_reason: str
    summary_table_text: str


class ModelComparer:
    """Compares CNN vs Random Forest and selects the best model."""

    @staticmethod
    def compare(
        cnn_metrics: Optional[ModelMetrics],
        rf_metrics: Optional[ModelMetrics],
    ) -> ComparisonResult:
        if cnn_metrics is None and rf_metrics is None:
            raise ValueError("No models provided for comparison.")

        if cnn_metrics is None:
            table = ModelComparer._format_single_table(rf_metrics)
            print(table)
            return ComparisonResult(
                winning_model_name=rf_metrics.model_name,
                winning_model_type="RandomForest",
                margin_accuracy=0.0,
                margin_f1=0.0,
                selection_reason="Only Random Forest model was trained.",
                summary_table_text=table,
            )

        if rf_metrics is None:
            table = ModelComparer._format_single_table(cnn_metrics)
            print(table)
            return ComparisonResult(
                winning_model_name=cnn_metrics.model_name,
                winning_model_type="CNN",
                margin_accuracy=0.0,
                margin_f1=0.0,
                selection_reason="Only CNN model was trained.",
                summary_table_text=table,
            )

        # Both models available: Compare on unseen test set
        diff_acc = (cnn_metrics.accuracy - rf_metrics.accuracy) * 100
        diff_f1 = (cnn_metrics.f1_macro - rf_metrics.f1_macro) * 100

        # Selection criteria: Macro F1 is primary (accounts for class balance), Accuracy is tie-breaker
        if cnn_metrics.f1_macro > rf_metrics.f1_macro:
            winner_name = cnn_metrics.model_name
            winner_type = "CNN"
            reason = f"CNN achieved higher Macro F1 (+{diff_f1:.1f}%) and Test Accuracy (+{diff_acc:.1f}%)."
        elif rf_metrics.f1_macro > cnn_metrics.f1_macro:
            winner_name = rf_metrics.model_name
            winner_type = "RandomForest"
            reason = f"Random Forest achieved higher Macro F1 (+{-diff_f1:.1f}%) and Test Accuracy (+{-diff_acc:.1f}%)."
        else:
            # Tied F1: use Accuracy
            if cnn_metrics.accuracy >= rf_metrics.accuracy:
                winner_name = cnn_metrics.model_name
                winner_type = "CNN"
                reason = "Tied F1 score; CNN selected based on test accuracy."
            else:
                winner_name = rf_metrics.model_name
                winner_type = "RandomForest"
                reason = "Tied F1 score; Random Forest selected based on test accuracy."

        table = ModelComparer._format_dual_table(cnn_metrics, rf_metrics, winner_name)
        print(table)

        return ComparisonResult(
            winning_model_name=winner_name,
            winning_model_type=winner_type,
            margin_accuracy=abs(diff_acc),
            margin_f1=abs(diff_f1),
            selection_reason=reason,
            summary_table_text=table,
        )

    @staticmethod
    def _format_dual_table(cnn: ModelMetrics, rf: ModelMetrics, winner: str) -> str:
        lines = [
            "\n" + "=" * 80,
            " 🏆  MODEL PERFORMANCE COMPARISON (ON UNSEEN TEST SET)",
            "=" * 80,
            f"{'Metric':<28} | {'Image-based CNN':<20} | {'Landmark Random Forest':<24} | {'Winner':<10}",
            "-" * 80,
            f"{'Test Accuracy':<28} | {cnn.accuracy * 100:>18.2f}% | {rf.accuracy * 100:>22.2f}% | {'CNN' if cnn.accuracy >= rf.accuracy else 'RF':<10}",
            f"{'Precision (Macro)':<28} | {cnn.precision_macro * 100:>18.2f}% | {rf.precision_macro * 100:>22.2f}% | {'CNN' if cnn.precision_macro >= rf.precision_macro else 'RF':<10}",
            f"{'Recall (Macro)':<28} | {cnn.recall_macro * 100:>18.2f}% | {rf.recall_macro * 100:>22.2f}% | {'CNN' if cnn.recall_macro >= rf.recall_macro else 'RF':<10}",
            f"{'F1-Score (Macro)':<28} | {cnn.f1_macro * 100:>18.2f}% | {rf.f1_macro * 100:>22.2f}% | {'CNN' if cnn.f1_macro >= rf.f1_macro else 'RF':<10}",
            f"{'F1-Score (Weighted)':<28} | {cnn.f1_weighted * 100:>18.2f}% | {rf.f1_weighted * 100:>22.2f}% | {'CNN' if cnn.f1_weighted >= rf.f1_weighted else 'RF':<10}",
            "-" * 80,
            f"  🎯 Overall Winner : {winner}",
            "=" * 80 + "\n",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_single_table(metrics: ModelMetrics) -> str:
        lines = [
            "\n" + "=" * 65,
            f" 🏆  EVALUATION SUMMARY: {metrics.model_name.upper()}",
            "=" * 65,
            f"  • Test Accuracy       : {metrics.accuracy * 100:.2f}%",
            f"  • Precision (Macro)   : {metrics.precision_macro * 100:.2f}%",
            f"  • Recall (Macro)      : {metrics.recall_macro * 100:.2f}%",
            f"  • F1-Score (Macro)    : {metrics.f1_macro * 100:.2f}%",
            f"  • F1-Score (Weighted) : {metrics.f1_weighted * 100:.2f}%",
            "=" * 65 + "\n",
        ]
        return "\n".join(lines)
