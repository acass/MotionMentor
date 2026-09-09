"""Unit tests for FeedbackGenerator plain-language coaching recommendations."""

import numpy as np
import pandas as pd
import pytest

from motion_mentor.comparison.feedback import FeedbackGenerator
from motion_mentor.storage.models import ComponentScores


@pytest.fixture
def base_dataframes():
    num_frames = 50
    timestamps = np.linspace(0, 2000, num_frames)

    ref_dict = {
        "frame_index": np.arange(num_frames),
        "timestamp_ms": timestamps,
        "index_mcp_mean": np.full(num_frames, 160.0),
        "index_mcp_std": np.full(num_frames, 4.0),
        "palm_pitch_mean": np.full(num_frames, 45.0),
        "palm_pitch_std": np.full(num_frames, 5.0),
    }
    ref_df = pd.DataFrame(ref_dict)

    trainee_dict = {
        "frame_index": np.arange(num_frames),
        "timestamp_ms": timestamps,
        "index_mcp": np.full(num_frames, 160.0),
        "palm_pitch": np.full(num_frames, 45.0),
    }
    trainee_df = pd.DataFrame(trainee_dict)
    return ref_df, trainee_df


def test_excellent_execution_feedback(base_dataframes):
    ref_df, trainee_df = base_dataframes
    generator = FeedbackGenerator()

    scores = ComponentScores(
        pose=99.0, trajectory=99.0, orientation=99.0, timing=99.0, smoothness=99.0, sequence=100.0
    )
    per_feature_z = {"index_mcp": 0.1, "palm_pitch": 0.2}

    items = generator.generate_feedback(
        component_scores=scores,
        per_feature_z=per_feature_z,
        aligned_ref_df=ref_df,
        aligned_trainee_df=trainee_df,
    )

    assert len(items) == 1
    assert items[0].component == "Overall"
    assert items[0].severity == "low"
    assert "Excellent execution" in items[0].message


def test_peak_error_detection_and_formatting(base_dataframes):
    ref_df, trainee_df = base_dataframes
    # Inject large deviation at frame 25 (t = 1.0s)
    trainee_df.loc[25, "index_mcp"] = 120.0  # 40 deg flexion deviation (10 sigma)

    generator = FeedbackGenerator()
    scores = ComponentScores(
        pose=75.0, trajectory=95.0, orientation=95.0, timing=95.0, smoothness=95.0, sequence=100.0
    )
    per_feature_z = {"index_mcp": 2.5}

    items = generator.generate_feedback(
        component_scores=scores,
        per_feature_z=per_feature_z,
        aligned_ref_df=ref_df,
        aligned_trainee_df=trainee_df,
    )

    assert len(items) >= 1
    top_item = items[0]
    assert top_item.component == "Pose"
    assert top_item.severity == "high"
    assert top_item.time_sec == pytest.approx(1.0, abs=0.1)
    assert "Index finger base (MCP)" in top_item.message
    assert "Recommendation:" not in top_item.message  # Recommendation is in separate field
    assert len(top_item.recommendation) > 0


def test_critical_failure_priority(base_dataframes):
    ref_df, trainee_df = base_dataframes
    generator = FeedbackGenerator()

    scores = ComponentScores(
        pose=90.0, trajectory=90.0, orientation=90.0, timing=90.0, smoothness=90.0, sequence=50.0
    )
    per_feature_z = {"index_mcp": 1.5}
    critical_failures = ["Critical checkpoint failed: 90° Wrist Rotation"]

    items = generator.generate_feedback(
        component_scores=scores,
        per_feature_z=per_feature_z,
        aligned_ref_df=ref_df,
        aligned_trainee_df=trainee_df,
        critical_failures=critical_failures,
    )

    assert len(items) >= 2
    # Critical failure must be the very first item
    assert items[0].component == "Sequence"
    assert items[0].severity == "high"
    assert "Critical checkpoint failed: 90° Wrist Rotation" in items[0].message
