"""MediaPipe Hand Tracking integration and skeleton visualization."""

from __future__ import annotations

import logging
import time
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from motion_mentor.storage.models import HandLandmarkData

logger = logging.getLogger(__name__)

# Standard 21 Hand Landmark Bones (connections)
HAND_CONNECTIONS = [
    # Palm / Base
    (0, 1),
    (1, 2),
    (2, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (17, 0),
    # Thumb
    (2, 3),
    (3, 4),
    # Index finger
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle finger
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky finger
    (17, 18),
    (18, 19),
    (19, 20),
]


class HandTracker:
    """Wrapper around MediaPipe Hands solution for 21-landmark tracking."""

    def __init__(
        self,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self.num_hands = num_hands
        self.static_image_mode = static_image_mode
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        timestamp_ms: float = 0.0,
    ) -> Tuple[List[HandLandmarkData], float]:
        """
        Process a single BGR OpenCV frame and extract hand landmarks.

        Returns:
            (hands_data, latency_ms): List of detected hands and inference latency.
        """
        t0 = time.perf_counter()

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.mp_hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        latency_ms = (time.perf_counter() - t0) * 1000.0

        hands_list: List[HandLandmarkData] = []
        if results and results.multi_hand_landmarks:
            num_detected = len(results.multi_hand_landmarks)
            for idx in range(num_detected):
                # Normalized image landmarks (0..1)
                img_lms = [
                    [float(lm.x), float(lm.y), float(lm.z)]
                    for lm in results.multi_hand_landmarks[idx].landmark
                ]

                # Metric 3D world landmarks (meters)
                world_lms: List[List[float]] = []
                if results.multi_hand_world_landmarks and idx < len(results.multi_hand_world_landmarks):
                    world_lms = [
                        [float(lm.x), float(lm.y), float(lm.z)]
                        for lm in results.multi_hand_world_landmarks[idx].landmark
                    ]

                # Handedness label and confidence score
                label = "Unknown"
                score = 0.0
                if results.multi_handedness and idx < len(results.multi_handedness):
                    classification = results.multi_handedness[idx].classification[0]
                    label = classification.label  # "Left" or "Right"
                    score = float(classification.score)

                hands_list.append(
                    HandLandmarkData(
                        hand_track_id=idx,
                        handedness=label,  # type: ignore
                        handedness_score=score,
                        landmarks_image=img_lms,
                        landmarks_world=world_lms,
                        valid=True,
                    )
                )

        return hands_list, latency_ms

    def close(self) -> None:
        """Release underlying landmarker resources."""
        if hasattr(self, "mp_hands") and self.mp_hands is not None:
            self.mp_hands.close()


def draw_hand_skeleton(
    image: np.ndarray,
    hands: List[HandLandmarkData],
    show_labels: bool = True,
    line_thickness: int = 2,
    joint_radius: int = 4,
) -> np.ndarray:
    """
    Render hand skeleton, joints, and handedness label onto an OpenCV image.
    Modifies and returns the image.
    """
    h, w = image.shape[:2]

    # Color palette (BGR)
    # Right hand: Cyan/Emerald bones, Yellow joints
    # Left hand: Orange/Coral bones, Magenta joints
    colors = {
        "Right": {
            "bone": (60, 220, 100),
            "joint": (0, 240, 255),
            "text": (0, 255, 128),
        },
        "Left": {
            "bone": (240, 140, 40),
            "joint": (255, 60, 180),
            "text": (255, 160, 60),
        },
        "Unknown": {
            "bone": (180, 180, 180),
            "joint": (240, 240, 240),
            "text": (200, 200, 200),
        },
    }

    for hand in hands:
        if not hand.landmarks_image or len(hand.landmarks_image) < 21:
            continue

        c = colors.get(hand.handedness, colors["Unknown"])

        # Convert normalized coordinates [0..1] to pixel positions [px, py]
        pixel_pts: List[Tuple[int, int]] = []
        for pt in hand.landmarks_image:
            px = int(np.clip(pt[0] * w, 0, w - 1))
            py = int(np.clip(pt[1] * h, 0, h - 1))
            pixel_pts.append((px, py))

        # Draw bones / connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = pixel_pts[start_idx]
            pt2 = pixel_pts[end_idx]
            cv2.line(image, pt1, pt2, c["bone"], line_thickness, cv2.LINE_AA)

        # Draw joints
        for idx, (px, py) in enumerate(pixel_pts):
            radius = joint_radius + 1 if idx in (0, 4, 8, 12, 16, 20) else joint_radius
            cv2.circle(image, (px, py), radius, c["joint"], -1, cv2.LINE_AA)

        # Draw handedness label at wrist (landmark 0)
        if show_labels and pixel_pts:
            wx, wy = pixel_pts[0]
            label_text = f"{hand.handedness} ({int(hand.handedness_score * 100)}%)"
            ty = max(20, wy - 12)
            cv2.putText(
                image,
                label_text,
                (wx - 20, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label_text,
                (wx - 20, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                c["text"],
                1,
                cv2.LINE_AA,
            )

    return image
