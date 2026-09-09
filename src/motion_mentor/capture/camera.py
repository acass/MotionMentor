"""Camera capture abstraction supporting physical webcams and synthetic sources."""

from __future__ import annotations

import logging
import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def enumerate_cameras(max_devices_to_check: int = 4) -> List[int]:
    """Scan and return a list of available camera device indices."""
    available = []
    for idx in range(max_devices_to_check):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


class CameraCapture:
    """OpenCV VideoCapture wrapper with monotonic timestamping and drop detection.

    Frame sources (this and SyntheticCamera) share the duck-typed interface:
    ``read() -> (success, frame_bgr, monotonic_timestamp_ms)`` and ``release()``.
    """

    def __init__(
        self,
        device_id: int | str = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        buffer_size: int = 1,
    ) -> None:
        self.device_id = device_id
        self.requested_width = width
        self.requested_height = height
        self.target_fps = target_fps
        self.expected_frame_interval_ms = 1000.0 / target_fps

        # Initialize capture
        if isinstance(device_id, str) and not device_id.isdigit():
            self.cap = cv2.VideoCapture(device_id)
        else:
            self.cap = cv2.VideoCapture(int(device_id))

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera device: {device_id}")

        # Set capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        # Actual hardware properties
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or float(target_fps)

        # Telemetry state
        self.frame_count = 0
        self.dropped_frame_count = 0
        self.last_timestamp_ms = 0.0
        self.start_time_ms = 0.0

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        ret, frame = self.cap.read()
        ts_ms = time.monotonic() * 1000.0

        if not ret or frame is None:
            return False, None, ts_ms

        if self.frame_count == 0:
            self.start_time_ms = ts_ms
        else:
            delta_ms = ts_ms - self.last_timestamp_ms
            # If delta is more than 1.8x expected frame interval, count dropped frame
            if delta_ms > (self.expected_frame_interval_ms * 1.8):
                dropped = int(delta_ms / self.expected_frame_interval_ms) - 1
                self.dropped_frame_count += max(1, dropped)

        self.last_timestamp_ms = ts_ms
        self.frame_count += 1
        return True, frame, ts_ms

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()


class SyntheticCamera:
    """
    Synthetic camera generator for testing without physical hardware.
    Renders an animated test pattern with a moving hand-like marker.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        total_seconds: float = 60.0,
    ) -> None:
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.expected_frame_interval_ms = 1000.0 / target_fps
        self.total_frames = int(total_seconds * target_fps)
        self.frame_count = 0
        self.start_time = time.monotonic()
        self.last_time_ms = 0.0
        self.dropped_frame_count = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self.frame_count >= self.total_frames:
            return False, None, self.last_time_ms

        ts_ms = (time.monotonic() - self.start_time) * 1000.0
        self.last_time_ms = ts_ms

        # Generate sleek dark synthetic background
        frame = np.full((self.height, self.width, 3), 28, dtype=np.uint8)

        # Draw grid
        grid_color = (42, 42, 42)
        for x in range(0, self.width, 80):
            cv2.line(frame, (x, 0), (x, self.height), grid_color, 1)
        for y in range(0, self.height, 80):
            cv2.line(frame, (0, y), (self.width, y), grid_color, 1)

        # Animated synthetic hand marker
        t = self.frame_count / float(self.target_fps)
        cx = int(self.width * 0.5 + 200 * math.sin(t * 1.5))
        cy = int(self.height * 0.5 + 100 * math.cos(t * 1.5))

        # Palm circle
        cv2.circle(frame, (cx, cy), 45, (180, 140, 100), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 45, (220, 200, 180), 2, cv2.LINE_AA)

        # 5 fingers
        angles = [-0.6, -0.3, 0.0, 0.3, 0.6]
        lengths = [65, 85, 95, 85, 70]
        for a, length in zip(angles, lengths):
            fx = int(cx + length * math.sin(a + math.sin(t * 2) * 0.1))
            fy = int(cy - length * math.cos(a + math.sin(t * 2) * 0.1))
            cv2.line(frame, (cx, cy), (fx, fy), (200, 160, 120), 12, cv2.LINE_AA)
            cv2.circle(frame, (fx, fy), 8, (240, 220, 200), -1, cv2.LINE_AA)

        # Label synthetic source
        cv2.putText(
            frame,
            f"SYNTHETIC CAMERA FEED | Frame {self.frame_count:04d} | t={t:.2f}s",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        self.frame_count += 1
        return True, frame, ts_ms

    def release(self) -> None:
        pass
