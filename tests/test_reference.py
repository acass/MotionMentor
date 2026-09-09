"""Unit tests for ReferenceProfileBuilder and tolerance envelope calculations."""

import numpy as np
import pandas as pd
import pytest

from motion_mentor.comparison.reference import ReferenceProfileBuilder, get_feature_min_tolerance
from motion_mentor.storage.models import Activity, Session, generate_uuid


@pytest.fixture
def sample_activity() -> Activity:
    return Activity(
        activity_id="reach_and_pinch",
        name="Reach and Pinch",
        description="Benchmark motion",
    )


def create_synthetic_expert_df(num_frames: int = 100, rot_offset: float = 0.0) -> pd.DataFrame:
    """Create a synthetic feature dataframe mimicking an expert reach and pinch."""
    timestamps = np.linspace(0, 3300, num_frames)
    t = np.linspace(0, np.pi, num_frames)

    data = {
        "frame_index": np.arange(num_frames),
        "timestamp_ms": timestamps,
        "valid": np.ones(num_frames, dtype=int),
        "wrist_x": 0.5 + 0.2 * np.sin(t),
        "wrist_y": 0.5 - 0.1 * np.cos(t),
        "wrist_z": -0.1 * np.sin(t),
        "index_mcp": 160.0 - 40.0 * np.sin(t),
        "index_pip": 170.0 - 50.0 * np.sin(t),
        "index_dip": 175.0 - 30.0 * np.sin(t),
        "thumb_mcp": 150.0 - 30.0 * np.sin(t),
        "thumb_ip": 160.0 - 40.0 * np.sin(t),
        "thumb_cmc": 140.0 - 20.0 * np.sin(t),
        "middle_mcp": 165.0 - 20.0 * np.sin(t),
        "middle_pip": 170.0 - 30.0 * np.sin(t),
        "middle_dip": 175.0 - 20.0 * np.sin(t),
        "ring_mcp": 165.0 - 15.0 * np.sin(t),
        "ring_pip": 170.0 - 20.0 * np.sin(t),
        "ring_dip": 175.0 - 15.0 * np.sin(t),
        "pinky_mcp": 165.0 - 10.0 * np.sin(t),
        "pinky_pip": 170.0 - 15.0 * np.sin(t),
        "pinky_dip": 175.0 - 10.0 * np.sin(t),
        "pinch_distance": 0.8 - 0.3 * np.sin(t),
        "palm_pitch": 10.0 + (50.0 + rot_offset) * np.sin(t),
        "palm_roll": 5.0 + 30.0 * np.sin(t),
        "palm_yaw": 0.0 + 5.0 * np.sin(t),
        "palm_normal_x": 0.0 + 0.3 * np.sin(t),
        "palm_normal_y": 0.0 + 0.3 * np.sin(t),
        "palm_normal_z": 1.0 - 0.2 * np.sin(t),
        "wrist_velocity": 0.1 * np.sin(t),
        "wrist_jerk": 0.05 * np.sin(t),
    }
    return pd.DataFrame(data)


def test_get_feature_min_tolerance():
    assert get_feature_min_tolerance("index_mcp") == 4.0
    assert get_feature_min_tolerance("pinch_distance") == 0.04
    assert get_feature_min_tolerance("wrist_x") == 0.03
    assert get_feature_min_tolerance("palm_normal_x") == 0.05
    assert get_feature_min_tolerance("wrist_velocity") == 0.1


def test_build_profile_single_demonstration(sample_activity, tmp_path):
    builder = ReferenceProfileBuilder(activity=sample_activity)
    sess = Session(
        session_id=generate_uuid(),
        activity_id=sample_activity.activity_id,
        user_id="expert_1",
        role="expert",
        duration_seconds=3.3,
    )
    df = create_synthetic_expert_df(num_frames=80)

    profile, envelope_df = builder.build_profile([(sess, df)], output_dir=tmp_path)

    assert profile.activity_id == sample_activity.activity_id
    assert profile.total_demonstrations == 1
    assert profile.medoid_session_id == sess.session_id
    assert len(envelope_df) == 80

    # Ensure mean equals original features and std is clamped to min tolerance
    np.testing.assert_allclose(envelope_df["index_mcp_mean"], df["index_mcp"], rtol=1e-5)
    assert (envelope_df["index_mcp_std"] >= 4.0).all()
    assert (envelope_df["pinch_distance_std"] >= 0.04).all()
    assert (envelope_df["wrist_x_std"] >= 0.03).all()


def test_build_profile_multiple_demonstrations_medoid(sample_activity, tmp_path):
    builder = ReferenceProfileBuilder(activity=sample_activity)

    sess1 = Session(session_id="sess_1", activity_id=sample_activity.activity_id, user_id="e1", role="expert", duration_seconds=3.3)
    df1 = create_synthetic_expert_df(num_frames=100, rot_offset=0.0)

    sess2 = Session(session_id="sess_2", activity_id=sample_activity.activity_id, user_id="e2", role="expert", duration_seconds=3.3)
    df2 = create_synthetic_expert_df(num_frames=100, rot_offset=5.0)

    sess3 = Session(session_id="sess_3", activity_id=sample_activity.activity_id, user_id="e3", role="expert", duration_seconds=3.3)
    df3 = create_synthetic_expert_df(num_frames=100, rot_offset=-5.0)

    takes = [(sess1, df1), (sess2, df2), (sess3, df3)]
    profile, envelope_df = builder.build_profile(takes, output_dir=tmp_path)

    assert profile.total_demonstrations == 3
    # sess1 is in the exact center (0.0 offset), so it should be the medoid
    assert profile.medoid_session_id == "sess_1"
    assert len(envelope_df) == 100

    # The std across takes should be positive and bounded by min tolerance
    assert (envelope_df["palm_pitch_std"] >= 4.0).all()
