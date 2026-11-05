"""
train_model.py
==============
Main entry point for the Sign Language Machine Learning Training Pipeline.

Trains both:
  1. Image-based Deep Convolutional Neural Network (PyTorch)
  2. MediaPipe Hand Landmark-based Random Forest Classifier (scikit-learn)

Then evaluates both models on an identical UNSEEN test set, prevents data leakage,
compares Accuracy, Precision, Recall, F1-Score, and Confusion Matrices, and saves
the best-performing model along with complete evaluation reports.

Usage:
  python train_model.py
  python train_model.py --epochs 20 --batch-size 16
  python train_model.py --skip-cnn        # Train only Random Forest
  python train_model.py --skip-rf         # Train only CNN
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Configure UTF-8 for Windows PowerShell / CMD
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from model_training.config import (
    DATASET_DIR,
    DEFAULT_EPOCHS,
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    LOGS_DIR,
    MODELS_DIR,
    EVALUATION_DIR,
)
from model_training.dataset_loader import DatasetLoader
from model_training.cnn_model import CNNTrainer, SignGestureCNN
from model_training.landmark_model import RandomForestSignModel
from model_training.evaluator import ModelEvaluator
from model_training.comparison import ModelComparer
from model_training.saver import ModelArtifactSaver

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "training_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("train_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign Language 26-Class Alphabet Recognition Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of training epochs for CNN (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size for CNN training (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--patience", type=int, default=EARLY_STOP_PATIENCE,
        help=f"Early stopping patience for CNN (default: {EARLY_STOP_PATIENCE})",
    )
    parser.add_argument(
        "--skip-cnn", action="store_true",
        help="Skip image-based CNN training and train only Random Forest.",
    )
    parser.add_argument(
        "--skip-rf", action="store_true",
        help="Skip landmark Random Forest training and train only CNN.",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default=str(DATASET_DIR),
        help="Path to the dataset directory containing alphabet folders A-Z.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Optional: Maximum total samples to balance and train on (e.g. 5200).",
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=220,
        help="Maximum balanced images per alphabet class (default: 220). Set 0 for all available images.",
    )
    parser.add_argument(
        "--n-trees", type=int, default=250,
        help="Number of decision trees for Random Forest (default: 250).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_start_time = time.time()

    print("\n" + "=" * 75)
    print(" 🤟 SIGN LANGUAGE MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 75)
    print(f"  Dataset Directory : {args.dataset_dir}")
    print(f"  Models Directory  : {MODELS_DIR}")
    print(f"  Evaluation Reports: {EVALUATION_DIR}")
    effective_max_samples = args.max_samples
    if effective_max_samples is None and args.samples_per_class and args.samples_per_class > 0:
        effective_max_samples = args.samples_per_class * 26

    if effective_max_samples:
        print(f"  Max Samples Limit : {effective_max_samples:,} (Balanced Subsampling, ~{args.samples_per_class or effective_max_samples//26}/class)")
    else:
        print("  Max Samples Limit : ALL available images")
    print("=" * 75)

    # ── Step 1: Load and Split Dataset (Leakage-Preventing) ──
    loader = DatasetLoader(dataset_dir=Path(args.dataset_dir))
    splits, distribution = loader.load_and_split(max_samples=effective_max_samples)

    if splits.total == 0:
        print("❌ Error: No valid images found in dataset directory.")
        print(f"   Please check {args.dataset_dir} or run capture_gestures.py to collect images.")
        sys.exit(1)

    if len(splits.test) == 0:
        print("❌ Error: Test split could not be created (insufficient data).")
        print("   Please add at least a few images to proceed.")
        sys.exit(1)

    # ── Step 2: Save Alphabet Class Labels ──
    saver = ModelArtifactSaver()
    classes_path = saver.save_class_labels()
    print(f"  📁 Class mapping saved → {classes_path}")

    evaluator = ModelEvaluator()
    cnn_metrics = None
    rf_metrics = None
    cnn_trainer = None
    rf_model_wrapper = None
    cnn_history = None

    # ── Step 3: Train Image-Based CNN Model ──
    if not args.skip_cnn:
        try:
            cnn_trainer = CNNTrainer(batch_size=args.batch_size)
            cnn_history = cnn_trainer.train_epochs(
                train_samples=splits.train,
                val_samples=splits.val,
                epochs=args.epochs,
                patience=args.patience,
            )

            # Evaluate on UNSEEN test set
            print("\n  🔬 Evaluating CNN on unseen test set...")
            y_test_true, y_test_pred, _ = cnn_trainer.predict_test_set(splits.test)
            cnn_metrics = evaluator.evaluate("Image-based CNN", y_test_true, y_test_pred)

            # Plot learning curves
            curve_path = evaluator.plot_cnn_learning_curves(cnn_history)
            if curve_path:
                print(f"  📊 Learning curves plot saved → {curve_path}")

            # Save individual CNN model
            saver.save_cnn_model(cnn_trainer.model, "model_cnn.pth")

        except Exception as e:
            logger.error("CNN training failed: %s", e, exc_info=True)
            print(f"⚠️ CNN Training encountered an error: {e}")

    # ── Step 4: Train Landmark-Based Random Forest Model ──
    if not args.skip_rf:
        try:
            rf_model_wrapper = RandomForestSignModel(n_estimators=args.n_trees)
            rf_model_wrapper.fit(train_samples=splits.train, val_samples=splits.val)

            # Evaluate on UNSEEN test set
            print("\n  🔬 Evaluating Random Forest on unseen test set...")
            y_test_true_rf, y_test_pred_rf, _ = rf_model_wrapper.predict_test_set(splits.test)
            rf_metrics = evaluator.evaluate("Landmark Random Forest", y_test_true_rf, y_test_pred_rf)

            # Save individual Random Forest model
            saver.save_rf_model(rf_model_wrapper.model, "model_rf.joblib")

        except Exception as e:
            logger.error("Random Forest training failed: %s", e, exc_info=True)
            print(f"⚠️ Random Forest Training encountered an error: {e}")
        finally:
            if rf_model_wrapper:
                rf_model_wrapper.close()

    # ── Step 5: Side-by-Side Model Comparison ──
    comparison = None
    if cnn_metrics is not None or rf_metrics is not None:
        comparison = ModelComparer.compare(cnn_metrics, rf_metrics)

        # ── Step 6: Save Best Model & Evaluation Reports ──
        best_model_path = saver.save_best_model(
            winner_type=comparison.winning_model_type,
            cnn_model=cnn_trainer.model if cnn_trainer else None,
            rf_model=rf_model_wrapper.model if rf_model_wrapper else None,
            comparison=comparison,
        )

        txt_rep, json_rep = saver.save_evaluation_report(
            cnn_metrics=cnn_metrics,
            rf_metrics=rf_metrics,
            comparison=comparison,
            history=cnn_history,
        )

        total_elapsed = time.time() - total_start_time
        print("\n" + "=" * 75)
        print(" 🎉  PIPELINE COMPLETE!")
        print("=" * 75)
        print(f"  🏆 Selected Best Model : {comparison.winning_model_name}")
        print(f"  💾 Best Model Saved To : {best_model_path}")
        print(f"  📄 Text Report Saved To: {txt_rep}")
        print(f"  📄 JSON Report Saved To: {json_rep}")
        print(f"  ⏱️ Total Elapsed Time  : {total_elapsed:.1f}s")
        print("=" * 75 + "\n")
    else:
        print("❌ No models were evaluated.")


if __name__ == "__main__":
    main()
