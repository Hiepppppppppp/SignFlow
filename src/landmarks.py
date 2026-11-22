"""MediaPipe extraction and 3D normalization for 21 hand landmarks."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp


class HandLandmarkExtractor:
    def __init__(self, static_image_mode: bool = True, max_num_hands: int = 1,
                 min_detection_confidence: float = 0.55, min_tracking_confidence: float = 0.55):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.last_handedness: str = "Unknown"

    @staticmethod
    def normalize_landmarks(hand_landmarks) -> np.ndarray:
        """
        Return 63 normalized 3D values.
        Wrist (0) is origin.
        Scaled using 3D palm size (wrist to middle-finger MCP), preventing division
        by zero when hand tilts toward camera.
        """
        points = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
        if points.shape != (21, 3):
            raise ValueError(f"Expected 21 landmarks, got {points.shape}")

        # Translate wrist to origin
        points -= points[0]

        # 3D scale: distance from wrist to middle-finger MCP (landmark 9)
        scale_3d = float(np.linalg.norm(points[9]))
        scale_2d = float(np.linalg.norm(points[9, :2]))

        # Use 2D scale if reasonably large (matches original ASL training data),
        # but smoothly blend into 3D scale if hand tilts toward camera (foreshortening).
        if scale_2d > 0.15:
            scale = scale_2d
        elif scale_3d > 1e-4:
            scale = scale_3d
        else:
            scale = float(np.max(np.linalg.norm(points, axis=1)))
            if scale < 1e-4:
                scale = 1.0

        points /= scale
        return points.reshape(-1).astype(np.float32)

    def extract(self, image_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[object]]:
        """Extract 63 normalized landmarks and MediaPipe raw result."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        result = self.hands.process(image_rgb)

        self.last_handedness = "Unknown"
        if not result or not result.multi_hand_landmarks:
            return None, result

        if result.multi_handedness:
            try:
                self.last_handedness = result.multi_handedness[0].classification[0].label
            except Exception:
                self.last_handedness = "Unknown"

        return self.normalize_landmarks(result.multi_hand_landmarks[0]), result

    def draw(self, image_bgr: np.ndarray, result) -> np.ndarray:
        """Draw hand skeleton with clear landmarks and connections."""
        if result and result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image_bgr,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return image_bgr

    def close(self):
        self.hands.close()
