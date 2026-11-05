"""
preprocessing/config.py
========================
All preprocessing knobs in one place.
Edit this file to tune augmentation, image sizes, and output paths.
Nothing else needs to change when you adjust these values.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS  (all relative to project root)
# ─────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).resolve().parent.parent   # C:\SignGesture
DATASET_DIR         = BASE_DIR / "dataset"                     # original images (read-only)
PROCESSED_DIR       = BASE_DIR / "processed_dataset"           # ALL output lands here
PROC_IMAGES_DIR     = PROCESSED_DIR / "images"                 # resized/normalised PNGs
PROC_LANDMARKS_DIR  = PROCESSED_DIR / "landmarks"              # per-image .npy landmark files
PROC_ARRAYS_DIR     = PROCESSED_DIR / "arrays"                 # combined .npz arrays
PROC_REPORTS_DIR    = PROCESSED_DIR / "reports"
CACHE_FILE          = PROCESSED_DIR / ".cache" / "processed_hashes.json"  # incremental-run cache
LOGS_DIR            = BASE_DIR / "logs"

# ─────────────────────────────────────────────────────────────
# ALPHABET CLASS DEFINITIONS
# ─────────────────────────────────────────────────────────────
ALPHABETS   = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # ['A' .. 'Z']
NUM_CLASSES = 26
CLASS_NAMES = ALPHABETS   # alias for model training code

# ─────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────

# Target size for CNN-based classification (H, W)
# 224×224  → MobileNetV2 / EfficientNet / ResNet50
# 64×64    → lightweight custom CNN
IMAGE_SIZE_CNN       = (224, 224)
IMAGE_SIZE_LITE      = (64,  64)

# Colour space sent to the CNN pipeline
# Options: "RGB" | "GRAY"
COLOR_MODE           = "RGB"

# Normalisation: pixel values divided by this, then optionally shifted
NORMALIZE_SCALE      = 255.0           # output range [0, 1]

# ─────────────────────────────────────────────────────────────
# AUGMENTATION  — safe for hand-gesture recognition
# ─────────────────────────────────────────────────────────────
# Augmentation is disabled by default for VALIDATION splits.
# The pipeline applies it only during training set generation.

AUGMENTATION_ENABLED   = True
AUGMENT_COPIES         = 3            # extra augmented copies per original image

# Rotation  — keep small; large rotations change gesture meaning
AUG_ROTATION_RANGE     = 15           # ± degrees

# Brightness  — simulate different lighting conditions
AUG_BRIGHTNESS_RANGE   = (0.70, 1.30) # multiply pixel values by this factor

# Zoom  — slight zoom-in/out (no flip — would mirror-reverse gestures)
AUG_ZOOM_RANGE         = 0.10         # ± fraction (0.10 = ±10%)

# Shift  — small positional jitter (fraction of image size)
AUG_SHIFT_RANGE        = 0.10         # ± fraction in X and Y

# Gaussian blur  — simulate slight camera defocus
AUG_BLUR_PROB          = 0.30         # probability of applying blur per copy
AUG_BLUR_KERNEL        = (3, 3)       # kernel size

# Salt-and-pepper noise  — simulate sensor noise
AUG_NOISE_PROB         = 0.20         # probability of adding noise per copy
AUG_NOISE_INTENSITY    = 0.01         # fraction of pixels to flip

# NEVER DO these — they change sign meaning:
#   ✗ Horizontal flip   (mirrors gestures)
#   ✗ Vertical flip     (upside-down hand = different sign)
#   ✗ Large rotation    (> ±20° is unsafe)
#   ✗ Extreme zoom-out  (hand may disappear from frame)

# ─────────────────────────────────────────────────────────────
# MEDIAPIPE LANDMARK EXTRACTION
# ─────────────────────────────────────────────────────────────
LANDMARK_ENABLED          = True      # set False to skip landmark extraction entirely
MP_MAX_NUM_HANDS          = 1         # we expect one hand per image
MP_MIN_DETECTION_CONF     = 0.5       # MediaPipe detection confidence threshold
MP_MIN_TRACKING_CONF      = 0.5

# Number of hand landmarks MediaPipe detects (fixed at 21)
NUM_LANDMARKS             = 21
LANDMARK_DIMS             = 3         # x, y, z  per landmark
LANDMARK_VECTOR_SIZE      = NUM_LANDMARKS * LANDMARK_DIMS   # 63

# Normalisation for landmarks:
#   "wrist"   — subtract wrist (point 0), scale by max distance
#   "bbox"    — normalise into the hand bounding-box
#   "none"    — keep raw [0,1] MediaPipe coordinates
LANDMARK_NORMALIZATION    = "wrist"

# If MediaPipe detects no hand, choose what to do:
#   "zeros"   — fill landmark vector with zeros (mark as undetected)
#   "skip"    — exclude the image from landmark dataset
LANDMARK_FALLBACK         = "zeros"

# ─────────────────────────────────────────────────────────────
# TRAIN / VALIDATION / TEST SPLIT  (used when building arrays)
# ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10          # must sum to 1.0
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────
# INCREMENTAL PROCESSING
# ─────────────────────────────────────────────────────────────
# Only newly added (or changed) images are processed on re-runs.
INCREMENTAL_MODE = True     # set False to force full reprocess
