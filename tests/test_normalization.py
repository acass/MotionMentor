"""Unit tests for coordinate normalization and canonical hand-frame alignment."""

import numpy as np
from motion_mentor.processing.normalization import (
    extract_global_trajectory,
    normalize_hand_landmarks,
    normalize_session_records,
)
from motion_mentor.storage.models import HandLandmarkData, LandmarkFrameRecord


def test_translation_and_scale_invariance() -> None:
    # Construct synthetic hand: wrist at (100, 200, 0), middle MCP at (100, 300, 0)
    # Palm length = 100
    hand_small = np.zeros((21, 3), dtype=np.float64)
    hand_small[0] = [100.0, 200.0, 0.0]  # Wrist
    hand_small[9] = [100.0, 300.0, 0.0]  # Middle MCP
    hand_small[5] = [80.0, 280.0, 0.0]   # Index MCP
    hand_small[17] = [120.0, 280.0, 0.0] # Pinky MCP

    # Same hand translated and scaled by 2.5x
    hand_large = hand_small * 2.5 + [50.0, -80.0, 10.0]

    canon_small, scale_small, _ = normalize_hand_landmarks(hand_small, align_rotation=True)
    canon_large, scale_large, _ = normalize_hand_landmarks(hand_large, align_rotation=True)

    # Scale ratio check
    assert np.isclose(scale_large / scale_small, 2.5, atol=1e-4)

    # Wrist should be at (0, 0, 0) for both
    np.testing.assert_allclose(canon_small[0], [0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(canon_large[0], [0.0, 0.0, 0.0], atol=1e-5)

    # Middle MCP should be at (0, 1, 0) for both
    np.testing.assert_allclose(canon_small[9], [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(canon_large[9], [0.0, 1.0, 0.0], atol=1e-5)

    # All canonical coordinates should match exactly regardless of translation and zoom!
    np.testing.assert_allclose(canon_small, canon_large, atol=1e-4)


def test_extract_global_trajectory() -> None:
    records = []
    for i in range(5):
        hand = HandLandmarkData(
            hand_track_id=0,
            handedness="Right",
            landmarks_image=[[float(i * 10), float(i * 20), 0.0] for _ in range(21)],
            valid=True,
        )
        records.append(
            LandmarkFrameRecord(
                session_id="test",
                frame_index=i,
                timestamp_ms=float(i * 33.3),
                hands=[hand],
                valid=True,
            )
        )

    traj = extract_global_trajectory(records)
    assert traj.shape == (5, 3)
    assert traj[0, 0] == 0.0
    assert traj[4, 0] == 40.0
    assert traj[4, 1] == 80.0
