"""Gesture predictor combining neural network model with landmark geometry."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch

try:
    from .model import LandmarkClassifier
    from .config import DEFAULT_MODEL_PATH
    from .gestures import detect_geometric_gesture, get_gesture_info, GestureInfo
except ImportError:
    from model import LandmarkClassifier
    from config import DEFAULT_MODEL_PATH
    from gestures import detect_geometric_gesture, get_gesture_info, GestureInfo


class GesturePredictor:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH, confidence_threshold: float = 0.60):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Train it first.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = list(checkpoint.get('class_names') or checkpoint.get('classes', []))
        input_size = checkpoint.get('input_size', 63)
        num_classes = checkpoint.get('num_classes', len(self.class_names))

        self.model = LandmarkClassifier(input_size, num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.confidence_threshold = confidence_threshold

    @torch.inference_mode()
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict hand gesture from 63 normalized landmarks.
        Returns:
            label: Gesture label code (e.g. 'Like', 'V', 'S', etc.) or 'Unknown'
            confidence: Probability score [0.0, 1.0]
        """
        if features is None or len(features) < 63:
            return 'Unknown', 0.0

        # Step 1: Geometric analyzer (instant, highly deterministic for primary poses)
        geom_label, geom_conf = detect_geometric_gesture(features)

        # Step 2: Neural network classifier
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device).reshape(1, -1)
        probabilities = torch.softmax(self.model(x), dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)
        nn_score = float(confidence.item())
        nn_label = self.class_names[int(index.item())] if int(index.item()) < len(self.class_names) else 'Unknown'

        # Fusion: If geometric analysis detected a definitive pose (e.g. Like, Peace, Fist)
        if geom_label is not None and geom_conf >= 0.90:
            # If NN already agrees or if geometry is confident on a distinctive gesture like 'Like'
            if geom_label == 'Like':
                return 'Like', max(geom_conf, nn_score)
            if nn_label == geom_label:
                return nn_label, max(geom_conf, nn_score)
            # If NN is uncertain, favor strong geometric detection
            if nn_score < 0.80:
                return geom_label, geom_conf

        if nn_score >= self.confidence_threshold:
            return nn_label, nn_score

        # Fallback to geometric if NN was below threshold
        if geom_label is not None and geom_conf >= 0.85:
            return geom_label, geom_conf

        return 'Unknown', nn_score

    def predict_details(self, features: np.ndarray) -> Tuple[str, float, GestureInfo]:
        """Return (label, confidence, gesture_info)."""
        label, conf = self.predict(features)
        info = get_gesture_info(label)
        return label, conf, info
