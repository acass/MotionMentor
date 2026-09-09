"""Unit tests for smoothing and short-gap interpolation."""

import numpy as np
from motion_mentor.processing.smoothing import (
    OneEuroFilter,
    interpolate_short_gaps,
)
from motion_mentor.storage.models import HandLandmarkData, LandmarkFrameRecord


def test_one_euro_filter_jitter_reduction() -> None:
    filt = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.007)
    # Steady signal with random noise
    true_val = 10.0
    noisy_vals = [true_val + np.random.normal(0, 0.5) for _ in range(50)]

    filtered_vals = []
    for i, val in enumerate(noisy_vals):
        f = filt.filter(np.array([val]), timestamp_sec=i * 0.033)
        filtered_vals.append(float(f[0]))

    # Variance of filtered signal should be significantly lower than noisy signal
    raw_var = np.var(noisy_vals[10:])
    filt_var = np.var(filtered_vals[10:])
    assert filt_var < raw_var


def test_interpolate_short_gap() -> None:
    # 5 frames: valid, missing, missing, valid, valid (gap duration ~66ms <= 150ms)
    records = []
    for i in range(5):
        if i in (1, 2):
            # Missing gap
            records.append(
                LandmarkFrameRecord(
                    session_id="test",
                    frame_index=i,
                    timestamp_ms=float(i * 33.33),
                    hands=[],
                    valid=False,
                )
            )
        else:
            # Valid hand with x = i * 0.1
            hand = HandLandmarkData(
                hand_track_id=0,
                handedness="Right",
                handedness_score=0.9,
                landmarks_image=[[float(i * 0.1), 0.5, 0.0] for _ in range(21)],
                valid=True,
            )
            records.append(
                LandmarkFrameRecord(
                    session_id="test",
                    frame_index=i,
                    timestamp_ms=float(i * 33.33),
                    hands=[hand],
                    valid=True,
                )
            )

    interpolated = interpolate_short_gaps(records, max_gap_ms=150.0)

    # All frames should now be valid
    assert all(r.valid for r in interpolated)
    assert len(interpolated[1].hands) == 1
    # Check linear interpolation value for frame 1: between 0.0 and 0.3
    x_val = interpolated[1].hands[0].landmarks_image[0][0]
    assert 0.05 < x_val < 0.25


def test_reject_long_gap() -> None:
    # Gap of 10 frames (333ms > 150ms)
    records = []
    for i in range(12):
        if 1 <= i <= 10:
            records.append(
                LandmarkFrameRecord(
                    session_id="test",
                    frame_index=i,
                    timestamp_ms=float(i * 33.33),
                    hands=[],
                    valid=False,
                )
            )
        else:
            hand = HandLandmarkData(
                hand_track_id=0,
                handedness="Right",
                handedness_score=0.9,
                landmarks_image=[[0.5, 0.5, 0.0] for _ in range(21)],
                valid=True,
            )
            records.append(
                LandmarkFrameRecord(
                    session_id="test",
                    frame_index=i,
                    timestamp_ms=float(i * 33.33),
                    hands=[hand],
                    valid=True,
                )
            )

    interpolated = interpolate_short_gaps(records, max_gap_ms=150.0)
    # Long gap frames should remain invalid
    assert not interpolated[5].valid
    assert len(interpolated[5].hands) == 0
