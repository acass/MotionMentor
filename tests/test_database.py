"""Unit tests for SQLite database management."""

import pytest
from pathlib import Path
from motion_mentor.storage.database import DatabaseManager
from motion_mentor.storage.models import Activity, Session, QualitySummary


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    db_file = tmp_path / "test_motion_mentor.db"
    return DatabaseManager(db_file)


def test_activity_crud(temp_db: DatabaseManager) -> None:
    act = Activity(
        activity_id="reach_pinch_01",
        name="reach_and_pinch",
        version=1,
        expected_hands=1,
        description="Reach and pinch activity",
    )
    temp_db.save_activity(act)

    retrieved = temp_db.get_activity("reach_pinch_01")
    assert retrieved is not None
    assert retrieved.name == "reach_and_pinch"
    assert retrieved.expected_hands == 1

    by_name = temp_db.get_activity_by_name("reach_and_pinch")
    assert by_name is not None
    assert by_name.activity_id == "reach_pinch_01"

    all_acts = temp_db.list_activities()
    assert len(all_acts) == 1


def test_session_crud(temp_db: DatabaseManager) -> None:
    act = Activity(activity_id="act_01", name="test_act")
    temp_db.save_activity(act)

    q = QualitySummary(
        total_frames=120,
        frames_with_hand=118,
        detection_coverage_pct=98.3,
        effective_fps=30.0,
        meets_criteria=True,
    )

    session = Session(
        session_id="sess_01",
        activity_id="act_01",
        role="expert",
        participant_id="expert_user",
        total_frames=120,
        duration_seconds=4.0,
        video_path="data/recordings/sess_01.mp4",
        landmark_path="data/landmarks/sess_01.parquet",
        quality_summary=q,
    )

    temp_db.save_session(session)

    retrieved = temp_db.get_session("sess_01")
    assert retrieved is not None
    assert retrieved.session_id == "sess_01"
    assert retrieved.role == "expert"
    assert retrieved.quality_summary is not None
    assert retrieved.quality_summary.detection_coverage_pct == 98.3

    # Filter sessions
    expert_sessions = temp_db.list_sessions(role="expert")
    assert len(expert_sessions) == 1
    trainee_sessions = temp_db.list_sessions(role="trainee")
    assert len(trainee_sessions) == 0
