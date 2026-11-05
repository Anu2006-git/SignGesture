"""
model_training/saver.py
=======================
Artifact serializer for trained models, alphabet class labels,
training history, and evaluation reports.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib

from .config import _TORCH_AVAILABLE
if _TORCH_AVAILABLE:
    import torch
else:
    torch = None

from .comparison import ComparisonResult
from .config import (
    ALPHABETS,
    EVALUATION_DIR,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    MODELS_DIR,
    NUM_CLASSES,
)
from .evaluator import ModelMetrics

logger = logging.getLogger(__name__)


class ModelArtifactSaver:
    """Saves all models, class labels, and evaluation artifacts."""

    def __init__(self, models_dir: Path = MODELS_DIR, eval_dir: Path = EVALUATION_DIR):
        self.models_dir = Path(models_dir)
        self.eval_dir = Path(eval_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def save_class_labels(self) -> Path:
        """Saves alphabet class labels to classes.json."""
        data = {
            "num_classes": NUM_CLASSES,
            "classes": ALPHABETS,
            "label_to_idx": LABEL_TO_IDX,
            "idx_to_label": {str(k): v for k, v in IDX_TO_LABEL.items()},
        }
        out_path = self.models_dir / "classes.json"
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved class labels to %s", out_path)
        return out_path

    def save_cnn_model(self, model, filename: str = "model_cnn.pth") -> Path:
        """Saves PyTorch CNN model or SpatialImageNet model."""
        if _TORCH_AVAILABLE and torch is not None and isinstance(model, torch.nn.Module):
            out_path = self.models_dir / filename
            torch.save(model.state_dict(), str(out_path))
        else:
            safe_name = filename.replace(".pth", ".joblib")
            out_path = self.models_dir / safe_name
            joblib.dump(model, str(out_path))
        logger.info("Saved image-based model to %s", out_path)
        return out_path

    def save_rf_model(self, rf_model, filename: str = "model_rf.joblib") -> Path:
        """Saves Scikit-Learn Random Forest model."""
        out_path = self.models_dir / filename
        joblib.dump(rf_model, str(out_path))
        logger.info("Saved Random Forest model to %s", out_path)
        return out_path

    def save_best_model(
        self,
        winner_type: str,
        cnn_model: Optional[object] = None,
        rf_model: Optional[object] = None,
        comparison: Optional[ComparisonResult] = None,
    ) -> Path:
        """Saves the winning model as the default 'best_model'."""
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "winning_model_type": winner_type,
            "selection_reason": comparison.selection_reason if comparison else "Single model trained",
        }

        if winner_type == "CNN" and cnn_model is not None:
            if _TORCH_AVAILABLE and torch is not None and isinstance(cnn_model, torch.nn.Module):
                best_model_path = self.models_dir / "best_model_cnn.pth"
                torch.save(cnn_model.state_dict(), str(best_model_path))
                manifest["model_file"] = "best_model_cnn.pth"
                manifest["framework"] = "PyTorch"
            else:
                best_model_path = self.models_dir / "best_model_image_net.joblib"
                joblib.dump(cnn_model, str(best_model_path))
                manifest["model_file"] = "best_model_image_net.joblib"
                manifest["framework"] = "scikit-learn (SpatialImageNet)"
        elif winner_type == "RandomForest" and rf_model is not None:
            best_model_path = self.models_dir / "best_model_rf.joblib"
            joblib.dump(rf_model, str(best_model_path))
            manifest["model_file"] = "best_model_rf.joblib"
            manifest["framework"] = "scikit-learn"
        else:
            best_model_path = self.models_dir / "best_model_manifest.json"

        # Write manifest
        manifest_path = self.models_dir / "best_model_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return best_model_path

    def save_evaluation_report(
        self,
        cnn_metrics: Optional[ModelMetrics],
        rf_metrics: Optional[ModelMetrics],
        comparison: Optional[ComparisonResult],
        history: Optional[Dict] = None,
    ) -> Tuple[Path, Path]:
        """Saves both human-readable text and structured JSON evaluation reports."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 1. Text Report ──
        lines = [
            "=" * 85,
            " SIGN LANGUAGE DETECTION — MACHINE LEARNING EVALUATION REPORT",
            f" Generated: {timestamp}",
            "=" * 85,
            "",
            "1. MODEL COMPARISON SUMMARY",
            "-" * 85,
        ]

        if comparison:
            lines.append(comparison.summary_table_text)
            lines.append(f"Selection Decision: {comparison.selection_reason}\n")

        if cnn_metrics:
            lines += [
                "2. CNN MODEL DETAILED EVALUATION (ON UNSEEN TEST SET)",
                "-" * 85,
                f"Accuracy           : {cnn_metrics.accuracy * 100:.2f}%",
                f"Precision (Macro)  : {cnn_metrics.precision_macro * 100:.2f}%",
                f"Recall (Macro)     : {cnn_metrics.recall_macro * 100:.2f}%",
                f"F1-Score (Macro)   : {cnn_metrics.f1_macro * 100:.2f}%",
                f"F1-Score (Weighted): {cnn_metrics.f1_weighted * 100:.2f}%\n",
                "Classification Report:",
                cnn_metrics.classification_report_text,
                "",
            ]

        if rf_metrics:
            lines += [
                "3. RANDOM FOREST (LANDMARK) DETAILED EVALUATION (ON UNSEEN TEST SET)",
                "-" * 85,
                f"Accuracy           : {rf_metrics.accuracy * 100:.2f}%",
                f"Precision (Macro)  : {rf_metrics.precision_macro * 100:.2f}%",
                f"Recall (Macro)     : {rf_metrics.recall_macro * 100:.2f}%",
                f"F1-Score (Macro)   : {rf_metrics.f1_macro * 100:.2f}%",
                f"F1-Score (Weighted): {rf_metrics.f1_weighted * 100:.2f}%\n",
                "Classification Report:",
                rf_metrics.classification_report_text,
                "",
            ]

        lines += ["=" * 85, "END OF REPORT", "=" * 85]

        txt_path = self.eval_dir / "evaluation_report.txt"
        txt_path.write_text("\n".join(lines), encoding="utf-8")

        # ── 2. JSON Report ──
        json_data = {
            "timestamp": timestamp,
            "comparison": {
                "winner_name": comparison.winning_model_name if comparison else None,
                "winner_type": comparison.winning_model_type if comparison else None,
                "reason": comparison.selection_reason if comparison else None,
            },
            "models": {},
        }

        if cnn_metrics:
            json_data["models"]["cnn"] = {
                "test_samples": cnn_metrics.total_test_samples,
                "accuracy": cnn_metrics.accuracy,
                "precision_macro": cnn_metrics.precision_macro,
                "recall_macro": cnn_metrics.recall_macro,
                "f1_macro": cnn_metrics.f1_macro,
                "f1_weighted": cnn_metrics.f1_weighted,
                "per_class_f1": cnn_metrics.per_class_f1,
            }

        if rf_metrics:
            json_data["models"]["random_forest"] = {
                "test_samples": rf_metrics.total_test_samples,
                "accuracy": rf_metrics.accuracy,
                "precision_macro": rf_metrics.precision_macro,
                "recall_macro": rf_metrics.recall_macro,
                "f1_macro": rf_metrics.f1_macro,
                "f1_weighted": rf_metrics.f1_weighted,
                "per_class_f1": rf_metrics.per_class_f1,
            }

        if history:
            json_data["cnn_training_history"] = history

        json_path = self.eval_dir / "evaluation_report.json"
        json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

        return txt_path, json_path
