"""Integration tests for HandTracker and skeleton visualization."""

import numpy as np
import cv2
from motion_mentor.tracking.hand_tracker import (
    HandTracker,
    draw_hand_skeleton,
)
from motion_mentor.storage.models import HandLandmarkData


def test_hand_tracker_inference() -> None:
    tracker = HandTracker(num_hands=2)
    # Create blank test frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    hands, latency_ms = tracker.process_frame(frame, timestamp_ms=0.0)
    assert isinstance(hands, list)
    assert latency_ms > 0.0
    tracker.close()


def test_skeleton_drawing() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_lms = [[0.5 + 0.01 * i, 0.5 + 0.01 * i, 0.0] for i in range(21)]
    hand = HandLandmarkData(
        hand_track_id=0,
        handedness="Right",
        handedness_score=0.99,
        landmarks_image=img_lms,
        landmarks_world=[],
        valid=True,
    )
    result = draw_hand_skeleton(frame, [hand], show_labels=True)
    assert result.shape == (720, 1280, 3)
    # Ensure pixels were drawn (not pure black anymore)
    assert np.any(result > 0)
