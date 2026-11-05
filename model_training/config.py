"""
model_training/config.py
=========================
Hyperparameters, directory paths, and settings for the Sign Language
Machine Learning Training Pipeline.
"""

import os
from pathlib import Path

# PyTorch engine check (enabled via USE_PYTORCH=1 or when policy permits)
_TORCH_AVAILABLE = False
if os.environ.get("USE_PYTORCH", "0") == "1":
    try:
        import torch
        _TORCH_AVAILABLE = True
    except Exception:
        _TORCH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# DIRECTORY PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent   # C:\SignGesture
DATASET_DIR     = BASE_DIR / "dataset"
MODELS_DIR      = BASE_DIR / "models"
EVALUATION_DIR  = MODELS_DIR / "evaluation"
LOGS_DIR        = BASE_DIR / "logs"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# TARGET CLASSES
# ─────────────────────────────────────────────────────────────
ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]   # ['A'..'Z']
NUM_CLASSES = 26
LABEL_TO_IDX = {letter: idx for idx, letter in enumerate(ALPHABETS)}
IDX_TO_LABEL = {idx: letter for idx, letter in enumerate(ALPHABETS)}

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# ─────────────────────────────────────────────────────────────
# DATA SPLIT & DATA LEAKAGE PREVENTION
# ─────────────────────────────────────────────────────────────
# Train / Validation / Test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────
# CNN MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE_CNN   = (128, 128)      # (H, W) 128x128 for efficient & accurate training
BATCH_SIZE       = 16
DEFAULT_EPOCHS   = 25
LEARNING_RATE    = 0.001
WEIGHT_DECAY     = 1e-4
DROPOUT_RATE     = 0.35
EARLY_STOP_PATIENCE = 7

# Device configuration (Auto CUDA GPU if available, else CPU)
if _TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"

# ImageNet normalization stats
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────
# RANDOM FOREST LANDMARK MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 250
RF_MAX_DEPTH    = None
RF_RANDOM_STATE = 42
RF_N_JOBS       = -1

# MediaPipe Hand Landmarks: 88-dimensional isotropic geometric feature vector
NUM_LANDMARKS       = 21
LANDMARK_DIMS       = 3
FEATURE_VECTOR_SIZE = 88  # 63 normalized coords + 5 tip-wrist + 10 tip-tip + 5 curl + 5 angles
