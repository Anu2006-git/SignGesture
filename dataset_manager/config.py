"""
Configuration settings for the Sign Language Dataset Management System.
All paths and thresholds are defined here — edit this file to customize behavior.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent   # C:\SignGesture
DATASET_DIR = BASE_DIR / "dataset"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR    = BASE_DIR / "logs"

# ─────────────────────────────────────────────────────────────
# DATASET STRUCTURE
# ─────────────────────────────────────────────────────────────
ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]   # ['A', 'B', ..., 'Z']

# Sub-folder naming convention (dataset_1, dataset_2, …)
DATASET_FOLDER_PREFIX = "dataset_"

# ─────────────────────────────────────────────────────────────
# SUPPORTED IMAGE FORMATS
# ─────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# ─────────────────────────────────────────────────────────────
# IMAGE VALIDATION THRESHOLDS
# ─────────────────────────────────────────────────────────────
MIN_IMAGE_WIDTH        = 32      # pixels — images smaller than this are flagged
MIN_IMAGE_HEIGHT       = 32
MAX_IMAGE_WIDTH        = 4096
MAX_IMAGE_HEIGHT       = 4096
MIN_FILE_SIZE_BYTES    = 1_000   # < 1 KB is suspicious
MAX_FILE_SIZE_BYTES    = 20_971_520  # 20 MB cap

# ─────────────────────────────────────────────────────────────
# DATA SUFFICIENCY THRESHOLDS
# ─────────────────────────────────────────────────────────────
MIN_IMAGES_WARNING   = 50    # yellow  — needs more data
MIN_IMAGES_GOOD      = 200   # green   — acceptable for training
MIN_IMAGES_EXCELLENT = 500   # blue    — excellent dataset

# ─────────────────────────────────────────────────────────────
# DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────
ENABLE_DUPLICATE_DETECTION = True   # hash-based; set False to skip (faster scans)

# ─────────────────────────────────────────────────────────────
# DASHBOARD DISPLAY
# ─────────────────────────────────────────────────────────────
PREVIEW_GRID_COLS     = 4    # columns in the image-preview grid
PREVIEW_MAX_IMAGES    = 8    # max preview images shown per alphabet
PREVIEW_THUMB_SIZE    = (120, 120)   # thumbnail dimensions (width, height)

# ─────────────────────────────────────────────────────────────
# REPORT SETTINGS
# ─────────────────────────────────────────────────────────────
REPORT_FILENAME = "dataset_summary_report.txt"
JSON_REPORT_FILENAME = "dataset_summary_report.json"
