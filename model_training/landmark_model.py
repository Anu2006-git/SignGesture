"""
model_training/landmark_model.py
================================
MediaPipe hand landmark feature extraction (88 dimensions) and Random Forest
classifier for Sign Language alphabet recognition.
"""

import hashlib
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import (
    FEATURE_VECTOR_SIZE,
    MODELS_DIR,
    NUM_CLASSES,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RF_N_JOBS,
    RF_RANDOM_STATE,
)
from .dataset_loader import ImageSample
from .feature_extractor import (
    FEATURE_SIZE,
    compute_class_statistics,
    extract_hand_features,
)

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    mp = None


class LandmarkExtractor:
    """
    Extracts 88-dimensional isotropic, scale-invariant, translation-invariant
    geometric features from images using MediaPipe Hands.
    
    Includes automatic margin padding to ensure hands touching image edges
    are accurately detected by MediaPipe.
    """

    def __init__(self, min_detection_confidence: float = 0.3):
        if not _MP_AVAILABLE or not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise ImportError(
                "MediaPipe with solutions.hands is required. Ensure mediapipe==0.10.14 is installed."
            )
        self.min_detection_confidence = min_detection_confidence
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
        )

    def extract_from_path(self, img_path: Path) -> Tuple[Optional[np.ndarray], bool]:
        """
        Reads image from disk, pads borders to guarantee MediaPipe palm detection,
        and extracts normalized 88-D feature vector.
        
        Returns:
            (feat_vector, True) if hand detected.
            (None, False) if hand NOT detected (never returns zeros!).
        """
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            return None, False

        # Attempt 1: Pad with constant black border (gives MediaPipe context for cropped images)
        padded = cv2.copyMakeBorder(bgr, 70, 70, 70, 70, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        h, w = padded.shape[:2]
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            feat = extract_hand_features(results.multi_hand_landmarks[0].landmark, w, h)
            return feat, True

        # Attempt 2: Fallback with larger padding (for very tight crops)
        padded2 = cv2.copyMakeBorder(bgr, 120, 120, 120, 120, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        h2, w2 = padded2.shape[:2]
        rgb2 = cv2.cvtColor(padded2, cv2.COLOR_BGR2RGB)
        results2 = self.hands.process(rgb2)

        if results2.multi_hand_landmarks:
            feat = extract_hand_features(results2.multi_hand_landmarks[0].landmark, w2, h2)
            return feat, True

        return None, False

    def extract_dataset(
        self,
        samples: List[ImageSample],
        cache_name: Optional[str] = None,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Extracts landmark feature vectors for an entire list of ImageSamples.
        Uses disk cache if available to enable sub-second reload.
        Only valid detected hand gestures are retained (no corrupted zero vectors).
        
        Returns:
            (X, y, detected_count)
        """
        cache_path = None
        if cache_name:
            cache_path = MODELS_DIR / f"cache_{cache_name}.joblib"
            if cache_path.exists():
                try:
                    cached = joblib.load(str(cache_path))
                    if cached.get("count_samples") == len(samples) and cached.get("version") == 2:
                        print(f"  ⚡ Loaded {len(cached['X']):,} pre-extracted landmark vectors from cache ({cache_path.name})")
                        return cached["X"], cached["y"], cached["detected"]
                except Exception as e:
                    logger.warning("Cache load failed, re-extracting: %s", e)

        X_list = []
        y_list = []
        detected_count = 0
        total = len(samples)
        t0 = time.time()

        for i, sample in enumerate(samples):
            vec, detected = self.extract_from_path(sample.path)
            if detected and vec is not None:
                X_list.append(vec)
                y_list.append(sample.label_idx)
                detected_count += 1

            if show_progress and (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                fps = (i + 1) / max(1e-3, elapsed)
                pct = detected_count / (i + 1) * 100
                print(f"    • Processed {i + 1:>5}/{total} images | Detected: {detected_count:>5} ({pct:.1f}%) | {fps:.1f} imgs/sec")

        X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, FEATURE_SIZE), dtype=np.float32)
        y = np.array(y_list, dtype=np.int32) if y_list else np.empty((0,), dtype=np.int32)

        if cache_path and len(X) > 0:
            try:
                joblib.dump({
                    "X": X,
                    "y": y,
                    "detected": detected_count,
                    "count_samples": total,
                    "version": 2,
                }, str(cache_path))
                print(f"  💾 Cached {len(X):,} landmark features to {cache_path.name}")
            except Exception as e:
                logger.warning("Failed to save feature cache: %s", e)

        return X, y, detected_count

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass


class RandomForestSignModel:
    """
    Random Forest Classifier trained on 88-D normalized MediaPipe hand landmark features.
    Computes and stores class centroid statistics for reliable Out-Of-Distribution (OOD)
    and Wrong Hand Gesture rejection.
    """

    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: Optional[int] = RF_MAX_DEPTH,
        random_state: int = RF_RANDOM_STATE,
        n_jobs: int = RF_N_JOBS,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.extractor = LandmarkExtractor()
        self.class_stats: Dict[int, Dict] = {}

    def fit(self, train_samples: List[ImageSample], val_samples: Optional[List[ImageSample]] = None) -> None:
        """Extracts features and fits the Random Forest model."""
        combined_train = list(train_samples)
        if val_samples:
            combined_train.extend(val_samples)

        print("\n" + "=" * 65)
        print(" 🌲  EXTRACTING 88-D LANDMARKS & TRAINING RANDOM FOREST MODEL")
        print(f"     Images: {len(combined_train):,} | Trees: {self.model.n_estimators}")
        print("=" * 65)

        X_train, y_train, detected = self.extractor.extract_dataset(
            combined_train,
            cache_name=f"train_{len(combined_train)}",
            show_progress=True,
        )
        detection_rate = (detected / max(1, len(combined_train))) * 100
        print(f"  Hand detection rate in training set: {detection_rate:.1f}% ({detected}/{len(combined_train)})")

        if len(X_train) == 0:
            raise ValueError("No hands detected in the training set.")

        print(f"  Training Random Forest with {self.model.n_estimators} trees on {len(X_train):,} samples...")
        t0 = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Calculate class statistics (centroids and threshold radii for Wrong Gesture detection)
        print("  Computing class boundary centroids for Wrong Hand Gesture detection...")
        self.class_stats = compute_class_statistics(X_train, y_train, num_classes=NUM_CLASSES)
        self.model.class_stats_ = self.class_stats
        self.model.feature_dim_ = FEATURE_SIZE
        self.model.model_version_ = 2

        print(f"  ✅ Random Forest training complete in {train_time:.1f}s.")

    def predict_test_set(self, test_samples: List[ImageSample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs evaluation on unseen test dataset.
        Returns: (y_true, y_pred, y_probs)
        """
        print("  Extracting landmarks for unseen test set...")
        X_test, y_test, detected = self.extractor.extract_dataset(
            test_samples,
            cache_name=f"test_{len(test_samples)}",
            show_progress=False,
        )
        detection_rate = (detected / max(1, len(test_samples))) * 100
        print(f"  Test set hand detection rate: {detection_rate:.1f}% ({detected}/{len(test_samples)})")

        if len(X_test) == 0:
            raise ValueError("No hands detected in the test set.")

        preds = self.model.predict(X_test)

        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(X_test)
            probs = np.zeros((len(y_test), NUM_CLASSES), dtype=np.float32)
            for idx, c in enumerate(self.model.classes_):
                if c < NUM_CLASSES:
                    probs[:, c] = raw_probs[:, idx]
        else:
            probs = np.zeros((len(y_test), NUM_CLASSES), dtype=np.float32)
            for i, p in enumerate(preds):
                probs[i, p] = 1.0

        return y_test, preds, probs

    def close(self):
        self.extractor.close()
