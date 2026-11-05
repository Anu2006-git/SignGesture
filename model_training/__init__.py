"""
model_training package
======================
Complete, independent Machine Learning training and evaluation pipeline
for Sign Language recognition (CNN + MediaPipe Random Forest).
"""

from .config import (
    ALPHABETS,
    DATASET_DIR,
    DEFAULT_EPOCHS,
    DEVICE,
    EVALUATION_DIR,
    MODELS_DIR,
    NUM_CLASSES,
)
from .dataset_loader import DatasetDistribution, DatasetLoader, DatasetSplits, ImageSample
from .cnn_model import CNNTrainer, SignGestureCNN, SignImageDataset
from .landmark_model import LandmarkExtractor, RandomForestSignModel
from .evaluator import ModelEvaluator, ModelMetrics
from .comparison import ComparisonResult, ModelComparer
from .saver import ModelArtifactSaver

__all__ = [
    "ALPHABETS",
    "NUM_CLASSES",
    "DATASET_DIR",
    "MODELS_DIR",
    "EVALUATION_DIR",
    "DEVICE",
    "DEFAULT_EPOCHS",
    "DatasetLoader",
    "DatasetSplits",
    "DatasetDistribution",
    "ImageSample",
    "SignGestureCNN",
    "SignImageDataset",
    "CNNTrainer",
    "LandmarkExtractor",
    "RandomForestSignModel",
    "ModelEvaluator",
    "ModelMetrics",
    "ModelComparer",
    "ComparisonResult",
    "ModelArtifactSaver",
]
