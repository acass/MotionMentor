"""Unit tests for Pydantic data models."""

from motion_mentor.storage.models import (
    Activity,
    HandLandmarkData,
    LandmarkFrameRecord,
    QualitySummary,
    Session,
)


def test_activity_defaults() -> None:
    act = Activity(name="reach_and_pinch")
    assert act.name == "reach_and_pinch"
    assert act.expected_hands == 1
    assert act.target_fps == 30
    assert "pose" in act.scoring_weights
    assert sum(act.scoring_weights.values()) == 1.0


def test_session_lifecycle() -> None:
    session = Session(
        activity_id="act-123",
        activity_name="reach_and_pinch",
        role="expert",
    )
    assert session.role == "expert"
    assert session.resolution == [1280, 720]
    assert session.nominal_fps == 30.0
    assert session.duration_seconds == 0.0


def test_hand_landmark_data() -> None:
    # 21 points
    img_lms = [[0.5, 0.5, 0.0] for _ in range(21)]
    world_lms = [[0.0, 0.0, 0.0] for _ in range(21)]
    hand = HandLandmarkData(
        hand_track_id=0,
        handedness="Right",
        handedness_score=0.98,
        landmarks_image=img_lms,
        landmarks_world=world_lms,
        valid=True,
    )
    assert hand.handedness == "Right"
    assert len(hand.landmarks_image) == 21


def test_landmark_frame_record() -> None:
    frame = LandmarkFrameRecord(
        session_id="sess-001",
        frame_index=10,
        timestamp_ms=333.3,
        hands=[],
        valid=False,
    )
    assert frame.frame_index == 10
    assert frame.valid is False


def test_quality_summary_thresholds() -> None:
    summary = QualitySummary(
        total_frames=100,
        frames_with_hand=95,
        detection_coverage_pct=95.0,
        dropped_frames=1,
        effective_fps=29.8,
        median_latency_ms=18.5,
        p95_latency_ms=24.0,
        meets_criteria=True,
    )
    assert summary.meets_criteria is True
    assert summary.detection_coverage_pct == 95.0
