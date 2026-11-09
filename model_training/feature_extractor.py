"""
model_training/feature_extractor.py
===================================
Robust, scale-invariant, translation-invariant, and aspect-ratio-invariant
MediaPipe hand-landmark feature extractor (88 dimensions) and
gesture verification / anomaly detection engine for Sign Language alphabet recognition.
"""

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Feature dimensions:
# 21 landmarks * 3 coords = 63 relative coords
# 5 fingertip-to-wrist distances
# 10 pairwise fingertip-to-fingertip distances
# 5 fingertip-to-MCP curl distances
# 5 joint angle cosines at PIP joints
# Total = 63 + 5 + 10 + 5 + 5 = 88 features
FEATURE_SIZE = 88


def extract_hand_features(landmarks, img_w: int = 640, img_h: int = 480) -> np.ndarray:
    """
    Extracts an 88-dimensional geometric feature vector from 21 MediaPipe hand landmarks.
    
    Invariant to:
      - Image resolution and aspect ratio (16:9, 4:3, square 1:1)
      - Hand translation / position in frame
      - Hand distance from camera (palm-length scale normalization)
    """
    # 1. Un-normalize coordinates into isotropic pixel space
    raw_pts = np.array(
        [[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in landmarks],
        dtype=np.float32,
    )

    # 2. Translate wrist (landmark 0) to origin
    wrist = raw_pts[0]
    rel_pts = raw_pts - wrist

    # 3. Reference scale: Anatomical palm length (Wrist 0 -> Middle finger MCP 9)
    # This distance is rigid and invariant to whether fingers are curled or open!
    palm_scale = float(np.linalg.norm(rel_pts[9]))
    if palm_scale < 1e-4:
        palm_scale = float(np.max(np.linalg.norm(rel_pts, axis=1)))
    if palm_scale < 1e-4:
        palm_scale = 1.0

    # 63 normalized relative coordinates
    rel_norm = (rel_pts / palm_scale).flatten()

    tips = [4, 8, 12, 16, 20]      # Thumb, Index, Middle, Ring, Pinky tips
    mcps = [2, 5, 9, 13, 17]       # Corresponding MCP/base joints

    # 5 fingertip-to-wrist distances
    d_wrist = [np.linalg.norm(rel_pts[t]) / palm_scale for t in tips]

    # 10 pairwise fingertip-to-fingertip distances (captures hand openness & finger spacing)
    d_tips = [
        np.linalg.norm(rel_pts[t1] - rel_pts[t2]) / palm_scale
        for t1, t2 in combinations(tips, 2)
    ]

    # 5 fingertip-to-MCP distances (captures individual finger curl/extension)
    d_mcp = [
        np.linalg.norm(rel_pts[t] - rel_pts[m]) / palm_scale
        for t, m in zip(tips, mcps)
    ]

    # 5 joint angle cosines at PIP joints (captures bending of each finger)
    # Joint triplets: (MCP, PIP, TIP)
    joints = [(1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    angles = []
    for a, b, c in joints:
        v1 = raw_pts[b] - raw_pts[a]
        v2 = raw_pts[c] - raw_pts[b]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cos_ang = float(np.dot(v1, v2) / denom)
        angles.append(float(np.clip(cos_ang, -1.0, 1.0)))

    feat = np.concatenate([rel_norm, d_wrist, d_tips, d_mcp, angles]).astype(np.float32)

    # Clean any NaN or Inf
    feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
    return feat


def compute_class_statistics(X: np.ndarray, y: np.ndarray, num_classes: int = 26) -> Dict[int, Dict[str, Any]]:
    """
    Computes class centroids and boundary thresholds in 88-D feature space
    for Out-Of-Distribution (OOD) / Wrong Hand Gesture detection.
    """
    class_stats = {}
    for c in range(num_classes):
        mask = (y == c)
        if not np.any(mask):
            continue
        samples = X[mask]
        centroid = np.mean(samples, axis=0)
        dists = np.linalg.norm(samples - centroid, axis=1)
        p95 = float(np.percentile(dists, 95))
        p99 = float(np.percentile(dists, 99))
        threshold_dist = float(max(p95 * 1.30, p99 * 1.15, 1.5))

        class_stats[c] = {
            "centroid": centroid.astype(np.float32),
            "threshold_dist": threshold_dist,
            "p95": p95,
            "p99": p99,
            "sample_count": int(len(samples)),
        }
    return class_stats


def evaluate_gesture(
    feat: np.ndarray,
    model: Any,
    class_stats: Optional[Dict[int, Dict[str, Any]]],
    classes: List[str],
    confidence_threshold: float = 0.35,
    min_margin: float = 0.03,
) -> Dict[str, Any]:
    """
    Evaluates an 88-D feature vector against the trained model and class statistics
    to classify the letter OR detect a WRONG / UNMATCHED hand gesture.
    """
    feat_2d = feat.reshape(1, -1)
    if hasattr(model, "n_jobs") and getattr(model, "n_jobs", None) != 1:
        try:
            model.n_jobs = 1
        except Exception:
            pass
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat_2d)[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        sorted_probs = np.sort(probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else 1.0
    else:
        top_idx = int(model.predict(feat_2d)[0])
        confidence = 1.0
        margin = 1.0
        probs = np.zeros(len(classes), dtype=np.float32)
        if top_idx < len(probs):
            probs[top_idx] = 1.0

    # Retrieve candidate letter
    candidate_letter = classes[top_idx] if top_idx < len(classes) else "?"

    # Distance to predicted class centroid
    dist = 0.0
    threshold_dist = 999.0
    if class_stats and top_idx in class_stats:
        centroid = class_stats[top_idx]["centroid"]
        threshold_dist = class_stats[top_idx]["threshold_dist"]
        dist = float(np.linalg.norm(feat - centroid))

    # Evaluate if this gesture genuinely matches a trained sign
    is_matched = True
    reason = "VALID"

    if confidence < confidence_threshold:
        is_matched = False
        reason = f"Low confidence ({confidence * 100:.1f}% < {confidence_threshold * 100:.0f}%)"
    elif margin < min_margin:
        is_matched = False
        reason = f"Ambiguous gesture (Margin {margin * 100:.1f}% < {min_margin * 100:.0f}%)"
    elif class_stats and dist > threshold_dist * 3.5 and confidence < 0.60:
        is_matched = False
        reason = f"Outlier shape (Distance {dist:.2f} > threshold {threshold_dist * 3.5:.2f})"

    return {
        "pred_letter": candidate_letter if is_matched else None,
        "candidate_letter": candidate_letter,
        "confidence": confidence,
        "margin": margin,
        "dist": dist,
        "threshold_dist": threshold_dist,
        "is_matched": is_matched,
        "reason": reason,
        "probs": probs,
    }
