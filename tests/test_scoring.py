"""Unit tests for ScoringEngine, tolerance-aware Z-score mapping, and checkpoint validation."""

import numpy as np
import pandas as pd
import pytest

from motion_mentor.comparison.scoring import (
    ScoringEngine,
    compute_component_zscore,
    determine_interpretation_band,
    zscore_to_score_100,
)
from motion_mentor.storage.models import Activity


@pytest.fixture
def reach_and_pinch_activity() -> Activity:
    return Activity(
        activity_id="reach_and_pinch",
        name="Reach and Pinch",
        description="Standard reach and pinch test",
        checkpoints=[
            {"id": "cp_rotation_reached", "name": "90° Wrist Rotation", "critical": True},
            {"id": "cp_pinch_contact", "name": "Thumb-Index Pinch Contact", "critical": True},
            {"id": "cp_hold_duration", "name": "Pinch Hold Duration (0.5s)", "critical": False},
        ],
    )


def create_mock_reference_envelope(num_frames: int = 100) -> pd.DataFrame:
    t = np.linspace(0, np.pi, num_frames)
    timestamps = np.linspace(0, 3000, num_frames)

    cols = {
        "frame_index": np.arange(num_frames),
        "timestamp_ms": timestamps,
        "wrist_x_mean": 0.5 + 0.1 * np.sin(t),
        "wrist_x_std": np.full(num_frames, 0.03),
        "wrist_y_mean": 0.5 - 0.1 * np.cos(t),
        "wrist_y_std": np.full(num_frames, 0.03),
        "wrist_z_mean": -0.1 * np.sin(t),
        "wrist_z_std": np.full(num_frames, 0.03),
        "palm_pitch_mean": 10.0 + 50.0 * np.sin(t),
        "palm_pitch_std": np.full(num_frames, 5.0),
        "palm_roll_mean": 5.0 + 30.0 * np.sin(t),
        "palm_roll_std": np.full(num_frames, 4.0),
        "palm_yaw_mean": np.zeros(num_frames),
        "palm_yaw_std": np.full(num_frames, 4.0),
        "palm_normal_x_mean": 0.3 * np.sin(t),
        "palm_normal_x_std": np.full(num_frames, 0.05),
        "palm_normal_y_mean": 0.3 * np.sin(t),
        "palm_normal_y_std": np.full(num_frames, 0.05),
        "palm_normal_z_mean": 1.0 - 0.2 * np.sin(t),
        "palm_normal_z_std": np.full(num_frames, 0.05),
        "pinch_distance_mean": 0.8 - 0.3 * np.sin(t),
        "pinch_distance_std": np.full(num_frames, 0.04),
        "wrist_velocity_mean": 0.1 * np.sin(t),
        "wrist_velocity_std": np.full(num_frames, 0.1),
        "wrist_jerk_mean": 0.05 * np.sin(t),
        "wrist_jerk_std": np.full(num_frames, 0.1),
    }

    # Add joint angles
    joint_angles = [
        "thumb_cmc", "thumb_mcp", "thumb_ip",
        "index_mcp", "index_pip", "index_dip",
        "middle_mcp", "middle_pip", "middle_dip",
        "ring_mcp", "ring_pip", "ring_dip",
        "pinky_mcp", "pinky_pip", "pinky_dip",
    ]
    for ja in joint_angles:
        cols[f"{ja}_mean"] = 160.0 - 30.0 * np.sin(t)
        cols[f"{ja}_std"] = np.full(num_frames, 4.0)

    return pd.DataFrame(cols)


def test_compute_component_zscore():
    trainee = np.array([10.0, 12.0, 14.0])
    mean = np.array([10.0, 10.0, 10.0])
    std = np.array([2.0, 2.0, 2.0])

    z = compute_component_zscore(trainee, mean, std)
    # deviations: 0, 2, 4 -> z: 0, 1, 2 -> mean = 1.0
    assert pytest.approx(z, 0.01) == 1.0


def test_zscore_to_score_100():
    assert zscore_to_score_100(0.0) == 100.0
    assert 70.0 <= zscore_to_score_100(1.0) <= 71.0
    assert 48.0 <= zscore_to_score_100(2.0) <= 51.0
    assert zscore_to_score_100(10.0) < 5.0


def test_evaluate_identical_attempt(reach_and_pinch_activity):
    ref_df = create_mock_reference_envelope(num_frames=100)
    engine = ScoringEngine(reach_and_pinch_activity)

    # Trainee identical to reference mean
    trainee_dict = {"frame_index": ref_df["frame_index"], "timestamp_ms": ref_df["timestamp_ms"]}
    for col in ref_df.columns:
        if col.endswith("_mean"):
            trainee_dict[col[:-5]] = ref_df[col]

    trainee_df = pd.DataFrame(trainee_dict)
    diagonal_path = [(i, i) for i in range(100)]

    scores, overall, critical_failures, _ = engine.evaluate_attempt(
        aligned_ref_df=ref_df,
        aligned_trainee_df=trainee_df,
        trainee_duration_sec=3.0,
        reference_duration_sec=3.0,
        warping_path=diagonal_path,
    )

    assert not critical_failures
    assert scores.pose >= 98.0
    assert scores.trajectory >= 98.0
    assert scores.orientation >= 98.0
    assert scores.timing >= 98.0
    assert scores.smoothness >= 98.0
    assert scores.sequence == 100.0
    assert overall >= 98.0
    assert determine_interpretation_band(overall, critical_failures) == "Excellent match"


def test_deliberate_wrist_rotation_error(reach_and_pinch_activity):
    ref_df = create_mock_reference_envelope(num_frames=100)
    engine = ScoringEngine(reach_and_pinch_activity)

    trainee_dict = {"frame_index": ref_df["frame_index"], "timestamp_ms": ref_df["timestamp_ms"]}
    for col in ref_df.columns:
        if col.endswith("_mean"):
            trainee_dict[col[:-5]] = ref_df[col].copy()

    # Introduce large wrist rotation error (flat palm instead of 60 deg pitch)
    trainee_dict["palm_pitch"] = np.zeros(100)
    trainee_dict["palm_roll"] = np.zeros(100)

    trainee_df = pd.DataFrame(trainee_dict)
    diagonal_path = [(i, i) for i in range(100)]

    scores, overall, critical_failures, _ = engine.evaluate_attempt(
        aligned_ref_df=ref_df,
        aligned_trainee_df=trainee_df,
        trainee_duration_sec=3.0,
        reference_duration_sec=3.0,
        warping_path=diagonal_path,
    )

    # Pose should remain high, but orientation drops significantly
    assert scores.pose >= 95.0
    assert scores.orientation < 50.0
    # Rotation checkpoint should fail and cap overall score <= 50.0
    assert any("rotation" in f.lower() for f in critical_failures)
    assert overall <= 50.0
    assert determine_interpretation_band(overall, critical_failures) == "Needs review"


def test_critical_checkpoint_failure_capping(reach_and_pinch_activity):
    ref_df = create_mock_reference_envelope(num_frames=100)
    engine = ScoringEngine(reach_and_pinch_activity)

    trainee_dict = {"frame_index": ref_df["frame_index"], "timestamp_ms": ref_df["timestamp_ms"]}
    for col in ref_df.columns:
        if col.endswith("_mean"):
            trainee_dict[col[:-5]] = ref_df[col].copy()

    # Never close pinch: pinch distance stays wide (1.5)
    trainee_dict["pinch_distance"] = np.full(100, 1.5)

    trainee_df = pd.DataFrame(trainee_dict)
    diagonal_path = [(i, i) for i in range(100)]

    scores, overall, critical_failures, _ = engine.evaluate_attempt(
        aligned_ref_df=ref_df,
        aligned_trainee_df=trainee_df,
        trainee_duration_sec=3.0,
        reference_duration_sec=3.0,
        warping_path=diagonal_path,
    )

    assert any("pinch" in f.lower() for f in critical_failures)
    assert overall <= 50.0
    assert determine_interpretation_band(overall, critical_failures) == "Needs review"


def test_determine_interpretation_band():
    assert determine_interpretation_band(95.0, []) == "Excellent match"
    assert determine_interpretation_band(85.0, []) == "Good match"
    assert determine_interpretation_band(75.0, []) == "Developing"
    assert determine_interpretation_band(65.0, []) == "Needs review"
    assert determine_interpretation_band(95.0, ["Failed critical checkpoint"]) == "Needs review"
