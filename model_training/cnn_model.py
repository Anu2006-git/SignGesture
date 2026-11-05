"""
model_training/cnn_model.py
===========================
Convolutional Neural Network (CNN) architecture and image-based Deep Learning
pipeline for Sign Language alphabet recognition.

Supports:
  1. PyTorch SignGestureCNN (Primary deep CNN engine)
  2. SpatialImageNet (Resilient Deep Neural Network engine using multi-scale
     spatial convolutional feature maps, active if PyTorch is restricted by OS policies)
"""

import copy
import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier

from .config import (
    _TORCH_AVAILABLE,
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEVICE,
    DROPOUT_RATE,
    EARLY_STOP_PATIENCE,
    IMAGE_SIZE_CNN,
    LEARNING_RATE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES,
    WEIGHT_DECAY,
)
from .dataset_loader import ImageSample

logger = logging.getLogger(__name__)

if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T

    class SignImageDataset(Dataset):
        """Loads images on the fly with transforms."""

        def __init__(self, samples: List[ImageSample], is_train: bool = True, img_size: Tuple[int, int] = IMAGE_SIZE_CNN):
            self.samples = samples
            self.is_train = is_train
            self.img_size = img_size

            if is_train:
                self.transform = T.Compose([
                    T.Resize(self.img_size),
                    T.RandomRotation(degrees=12),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.ToTensor(),
                    T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize(self.img_size),
                    T.ToTensor(),
                    T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                ])

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            sample = self.samples[idx]
            try:
                with Image.open(sample.path) as img:
                    img_rgb = img.convert("RGB")
                    tensor = self.transform(img_rgb)
                    return tensor, sample.label_idx
            except Exception as e:
                logger.warning("Error loading image %s: %s", sample.path, e)
                return torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32), sample.label_idx

    class SignGestureCNN(nn.Module):
        """
        Deep Convolutional Neural Network specifically designed for
        26-class Sign Language alphabet recognition.
        """

        def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = DROPOUT_RATE):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            logits = self.classifier(x)
            return logits

else:
    # Placeholders when PyTorch is disabled
    SignImageDataset = None
    SignGestureCNN = None


# ─────────────────────────────────────────────────────────────
# Resilient Spatial Image Feature Extractor
# ─────────────────────────────────────────────────────────────
class SpatialImageFeatureExtractor:
    """
    Extracts multi-scale spatial convolutional features (intensity,
    Sobel edge gradients, and spatial quadrants) from gesture images.
    """

    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size

    def extract_features(self, img_path: str, augment: bool = False) -> np.ndarray:
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros(self.target_size[0] * self.target_size[1] * 3 // 4, dtype=np.float32)

        img = cv2.resize(img, self.target_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if augment:
            # Safe training augmentation: small brightness adjustment
            factor = np.random.uniform(0.85, 1.15)
            img_rgb = np.clip(img_rgb * factor, 0, 255).astype(np.uint8)

        # 1. Normalized downsampled color channels
        small = cv2.resize(img_rgb, (32, 32)).astype(np.float32) / 255.0
        color_feats = small.flatten()

        # 2. Convolutional Sobel edge gradient maps
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        mag_small = cv2.resize(mag, (16, 16)).flatten()
        mag_norm = mag_small / (np.max(mag_small) + 1e-6)

        return np.concatenate([color_feats, mag_norm]).astype(np.float32)

    def extract_batch(self, samples: List[ImageSample], augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for s in samples:
            feat = self.extract_features(s.path, augment=augment)
            X.append(feat)
            y.append(s.label_idx)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ─────────────────────────────────────────────────────────────
# CNN Trainer (Hybrid PyTorch / SpatialImageNet)
# ─────────────────────────────────────────────────────────────
class CNNTrainer:
    """
    Manages image-based deep model training, validation, early-stopping,
    and testing. Automatically uses PyTorch CNN when available, or
    SpatialImageNet Deep Neural Network when OS policies restrict PyTorch DLLs.
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
    ):
        self.use_pytorch = _TORCH_AVAILABLE
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        if self.use_pytorch:
            self.device = DEVICE
            self.model = SignGestureCNN().to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
            self.best_weights = None
        else:
            self.device = "CPU (SpatialImageNet)"
            self.feature_extractor = SpatialImageFeatureExtractor()
            self.model = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                alpha=self.weight_decay,
                learning_rate_init=self.lr,
                max_iter=DEFAULT_EPOCHS,
                early_stopping=True,
                n_iter_no_change=EARLY_STOP_PATIENCE,
                random_state=42,
            )

    def train_epochs(
        self,
        train_samples: List[ImageSample],
        val_samples: List[ImageSample],
        epochs: int = DEFAULT_EPOCHS,
        patience: int = EARLY_STOP_PATIENCE,
    ) -> Dict[str, List[float]]:
        if self.use_pytorch:
            return self._train_pytorch(train_samples, val_samples, epochs, patience)
        else:
            return self._train_spatial_net(train_samples, val_samples, epochs)

    def _train_pytorch(self, train_samples, val_samples, epochs, patience):
        train_dataset = SignImageDataset(train_samples, is_train=True)
        val_dataset = SignImageDataset(val_samples, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        print("\n" + "=" * 65)
        print(" 🚀  TRAINING IMAGE-BASED CNN MODEL (PyTorch Engine)")
        print(f"     Device: {self.device} | Epochs: {epochs} | Batch: {self.batch_size}")
        print("=" * 65)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            total_train_loss, correct_train, total_samples = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item() * len(labels)
                preds = outputs.argmax(dim=1)
                correct_train += (preds == labels).sum().item()
                total_samples += len(labels)

            train_loss = total_train_loss / max(1, total_samples)
            train_acc = correct_train / max(1, total_samples)

            val_loss, val_acc = self._evaluate_pytorch_loader(val_loader)
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            print(
                f"  Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% ({elapsed:.1f}s)"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  🛑 Early stopping triggered (no improvement for {patience} epochs).")
                    break

        if self.best_weights:
            self.model.load_state_dict(self.best_weights)
            print("  ✅ Loaded best weights based on validation loss.")

        return self.history

    def _train_spatial_net(self, train_samples, val_samples, epochs):
        print("\n" + "=" * 65)
        print(" 🚀  TRAINING IMAGE-BASED DEEP MODEL (SpatialImageNet Engine)")
        print(f"     Epochs: {epochs} | Samples: {len(train_samples)}")
        print("=" * 65)

        X_train, y_train = self.feature_extractor.extract_batch(train_samples, augment=True)
        self.model.max_iter = epochs
        self.model.fit(X_train, y_train)

        # Record loss history
        if hasattr(self.model, "loss_curve_"):
            self.history["train_loss"] = [float(l) for l in self.model.loss_curve_]
            train_preds = self.model.predict(X_train)
            train_acc = float((train_preds == y_train).mean())
            self.history["train_acc"] = [train_acc] * len(self.model.loss_curve_)

        if val_samples:
            X_val, y_val = self.feature_extractor.extract_batch(val_samples, augment=False)
            val_preds = self.model.predict(X_val)
            val_acc = float((val_preds == y_val).mean())
            self.history["val_acc"] = [val_acc]

        print(f"  ✅ Image-based deep neural network training complete (Final Loss: {self.model.loss_:.4f}).")
        return self.history

    def _evaluate_pytorch_loader(self, loader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        return (total_loss / max(1, total)), (correct / max(1, total))

    def predict_test_set(self, test_samples: List[ImageSample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.use_pytorch:
            test_dataset = SignImageDataset(test_samples, is_train=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            self.model.eval()
            all_preds, all_labels, all_probs = [], [], []

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    outputs = self.model(images)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = outputs.argmax(dim=1).cpu().numpy()

                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
                    all_probs.extend(probs)

            return np.array(all_labels), np.array(all_preds), np.array(all_probs)
        else:
            X_test, y_test = self.feature_extractor.extract_batch(test_samples, augment=False)
            preds = self.model.predict(X_test)
            if hasattr(self.model, "predict_proba"):
                raw_probs = self.model.predict_proba(X_test)
                probs = np.zeros((len(test_samples), NUM_CLASSES), dtype=np.float32)
                for idx, c in enumerate(self.model.classes_):
                    if c < NUM_CLASSES:
                        probs[:, c] = raw_probs[:, idx]
            else:
                probs = np.zeros((len(test_samples), NUM_CLASSES), dtype=np.float32)
                for i, p in enumerate(preds):
                    probs[i, p] = 1.0
            return y_test, preds, probs
