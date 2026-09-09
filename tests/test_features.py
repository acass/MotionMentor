"""Unit tests for geometric and kinematic feature derivation."""

import numpy as np
from motion_mentor.processing.features import (
    compute_3d_angle,
    compute_palm_orientation,
    extract_frame_features,
    extract_session_features_df,
)
from motion_mentor.storage.models import HandLandmarkData, LandmarkFrameRecord


def test_compute_3d_angle() -> None:
    # 90-degree right angle
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])  # Vertex
    c = np.array([0.0, 1.0, 0.0])
    assert np.isclose(compute_3d_angle(a, b, c), 90.0)

    # 180-degree straight line
    c_straight = np.array([-1.0, 0.0, 0.0])
    assert np.isclose(compute_3d_angle(a, b, c_straight), 180.0)

    # 0-degree coincident ray
    assert np.isclose(compute_3d_angle(a, b, a), 0.0)


def test_palm_orientation_flat() -> None:
    pts = np.zeros((21, 3))
    pts[0] = [0.0, 0.0, 0.0]    # Wrist
    pts[9] = [0.0, 1.0, 0.0]    # Middle MCP (pointing up)
    pts[5] = [-0.5, 0.8, 0.0]   # Index MCP (left)
    pts[17] = [0.5, 0.8, 0.0]   # Pinky MCP (right)

    normal, pitch, roll, yaw = compute_palm_orientation(pts)
    # Normal should point in +Z direction
    assert np.isclose(normal[2], 1.0, atol=1e-3)
    assert np.isclose(normal[0], 0.0, atol=1e-3)
    assert np.isclose(normal[1], 0.0, atol=1e-3)


def test_extract_frame_features() -> None:
    pts = np.zeros((21, 3))
    pts[0] = [0.0, 0.0, 0.0]
    pts[9] = [0.0, 1.0, 0.0]
    # Pinch: thumb tip (4) and index tip (8) touching at [0.1, 0.8, 0.0]
    pts[4] = [0.1, 0.8, 0.0]
    pts[8] = [0.1, 0.8, 0.0]

    feats = extract_frame_features(pts, palm_scale=1.0)
    assert "index_mcp" in feats
    assert "thumb_cmc" in feats
    assert "pinch_distance" in feats
    assert np.isclose(feats["pinch_distance"], 0.0, atol=1e-5)


def test_extract_session_features_kinematics() -> None:
    records = []
    # 10 frames moving at constant velocity (1.0 unit per second)
    fps = 30.0
    dt_ms = 1000.0 / fps
    for i in range(10):
        t_sec = i / fps
        hand = HandLandmarkData(
            hand_track_id=0,
            handedness="Right",
            # Moving along X
            landmarks_image=[[float(t_sec), 0.5, 0.0] for _ in range(21)],
            valid=True,
        )
        records.append(
            LandmarkFrameRecord(
                session_id="test_kinematics",
                frame_index=i,
                timestamp_ms=float(i * dt_ms),
                hands=[hand],
                valid=True,
            )
        )

    df = extract_session_features_df(records)
    assert not df.empty
    assert "wrist_velocity" in df.columns
    assert "wrist_acceleration" in df.columns
    assert "wrist_jerk" in df.columns

    # Check steady state velocity (approx 1.0)
    steady_v = df["wrist_velocity"].iloc[2:-1]
    np.testing.assert_allclose(steady_v, 1.0, atol=0.05)
