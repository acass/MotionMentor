"""Trajectory smoothing and short-gap interpolation for landmark sequences."""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Tuple
import numpy as np

from motion_mentor.storage.models import HandLandmarkData, LandmarkFrameRecord


class LowPassFilter:
    """First-order low-pass filter."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.y: Optional[np.ndarray] = None

    def filter(self, x: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = alpha
        if self.y is None:
            self.y = np.array(x, dtype=np.float64, copy=True)
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y


class OneEuroFilter:
    """
    1€ filter for speed-adaptive jitter reduction and low-latency tracking.
    Reference: Casiez et al., "1 € Filter: A Simple Speed-based Low-pass Filter
    for Noisy Input in Human-Computer Interaction", CHI 2012.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        te = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + te / dt) if (dt > 0) else 1.0

    def filter(self, x: np.ndarray, timestamp_sec: Optional[float] = None) -> np.ndarray:
        x = np.array(x, dtype=np.float64)

        if self.last_time is None or timestamp_sec is None:
            dt = 1.0 / self.freq
        else:
            dt = max(1e-4, timestamp_sec - self.last_time)

        self.last_time = timestamp_sec

        # Compute derivative (speed)
        if self.x_filter.y is None:
            dx = np.zeros_like(x)
        else:
            dx = (x - self.x_filter.y) / dt

        # Filter derivative
        alpha_d = self._alpha(self.d_cutoff, dt)
        edx = self.dx_filter.filter(dx, alpha_d)

        # Dynamic cutoff based on speed
        cutoff = self.min_cutoff + self.beta * np.abs(edx)

        # Filter signal with dynamic cutoff
        alpha = self._alpha(cutoff, dt)
        return self.x_filter.filter(x, alpha)


def interpolate_short_gaps(
    records: List[LandmarkFrameRecord],
    max_gap_ms: float = 150.0,
) -> List[LandmarkFrameRecord]:
    """
    Interpolate missing landmark frames over short durations (< max_gap_ms).
    Gaps exceeding max_gap_ms remain invalid.
    """
    if len(records) < 2:
        return copy.deepcopy(records)

    output = copy.deepcopy(records)
    n = len(output)

    # Scan for gaps in hand detection (assuming primary hand track 0)
    i = 0
    while i < n:
        if not output[i].hands or not output[i].valid:
            # Found start of gap
            gap_start = i
            while i < n and (not output[i].hands or not output[i].valid):
                i += 1
            gap_end = i  # first valid frame after gap, or n

            # Check if bounded by valid frames on both sides
            if gap_start > 0 and gap_end < n:
                t_prev = output[gap_start - 1].timestamp_ms
                t_next = output[gap_end].timestamp_ms
                gap_duration_ms = t_next - t_prev

                if gap_duration_ms <= max_gap_ms:
                    # Linearly interpolate between gap_start - 1 and gap_end
                    prev_hand = output[gap_start - 1].hands[0]
                    next_hand = output[gap_end].hands[0]

                    prev_img = np.array(prev_hand.landmarks_image, dtype=np.float32)
                    next_img = np.array(next_hand.landmarks_image, dtype=np.float32)

                    prev_world = (
                        np.array(prev_hand.landmarks_world, dtype=np.float32)
                        if prev_hand.landmarks_world
                        else np.zeros_like(prev_img)
                    )
                    next_world = (
                        np.array(next_hand.landmarks_world, dtype=np.float32)
                        if next_hand.landmarks_world
                        else np.zeros_like(next_img)
                    )

                    for k in range(gap_start, gap_end):
                        t_curr = output[k].timestamp_ms
                        weight = (t_curr - t_prev) / gap_duration_ms
                        interp_img = (1.0 - weight) * prev_img + weight * next_img
                        interp_world = (1.0 - weight) * prev_world + weight * next_world

                        interp_hand = HandLandmarkData(
                            hand_track_id=prev_hand.hand_track_id,
                            handedness=prev_hand.handedness,
                            handedness_score=min(prev_hand.handedness_score, next_hand.handedness_score) * 0.9,
                            landmarks_image=interp_img.tolist(),
                            landmarks_world=interp_world.tolist(),
                            valid=True,
                        )
                        output[k].hands = [interp_hand]
                        output[k].valid = True
        else:
            i += 1

    return output


def smooth_landmark_records(
    records: List[LandmarkFrameRecord],
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    interpolate_gaps: bool = True,
    max_gap_ms: float = 150.0,
) -> List[LandmarkFrameRecord]:
    """
    Full preprocessing pipeline:
    1. Interpolate short gaps (< max_gap_ms).
    2. Apply 1€ filter across 21 landmarks (both image and world space).
    """
    if interpolate_gaps:
        records = interpolate_short_gaps(records, max_gap_ms=max_gap_ms)

    output: List[LandmarkFrameRecord] = []
    filter_img: Optional[OneEuroFilter] = None
    filter_world: Optional[OneEuroFilter] = None

    for rec in records:
        rec_copy = copy.deepcopy(rec)
        if rec.hands and rec.valid:
            hand = rec_copy.hands[0]
            img_arr = np.array(hand.landmarks_image, dtype=np.float32)
            ts_sec = rec.timestamp_ms / 1000.0

            if filter_img is None:
                filter_img = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
                filter_world = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

            smoothed_img = filter_img.filter(img_arr, ts_sec)
            hand.landmarks_image = smoothed_img.tolist()

            if hand.landmarks_world:
                world_arr = np.array(hand.landmarks_world, dtype=np.float32)
                smoothed_world = filter_world.filter(world_arr, ts_sec)
                hand.landmarks_world = smoothed_world.tolist()

        output.append(rec_copy)

    return output
