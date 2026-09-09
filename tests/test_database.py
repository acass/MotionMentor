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


def test_reference_profile_and_assessment_crud(temp_db: DatabaseManager) -> None:
    from motion_mentor.storage.models import ReferenceProfile, AssessmentResult, ComponentScores, FeedbackItem

    act = Activity(activity_id="reach_and_pinch", name="Reach and Pinch")
    temp_db.save_activity(act)

    profile = ReferenceProfile(
        reference_id="ref_001",
        activity_id="reach_and_pinch",
        activity_name="Reach and Pinch",
        version=1,
        expert_session_ids=["sess_01"],
        medoid_session_id="sess_01",
        total_demonstrations=1,
        duration_mean_sec=3.5,
        profile_path="data/references/ref_001_reference.parquet",
    )
    temp_db.save_reference_profile(profile)

    retrieved_ref = temp_db.get_reference_profile("ref_001")
    assert retrieved_ref is not None
    assert retrieved_ref.reference_id == "ref_001"
    assert retrieved_ref.total_demonstrations == 1

    latest_ref = temp_db.get_latest_reference_profile("reach_and_pinch")
    assert latest_ref is not None
    assert latest_ref.reference_id == "ref_001"

    # Trainee Session (required for foreign key)
    trainee_sess = Session(
        session_id="sess_trainee",
        activity_id="reach_and_pinch",
        role="trainee",
        participant_id="trainee_user",
        total_frames=90,
        duration_seconds=3.0,
    )
    temp_db.save_session(trainee_sess)

    # Assessment Result
    comp = ComponentScores(
        pose=88.5,
        trajectory=91.0,
        orientation=85.0,
        timing=82.0,
        smoothness=89.0,
        sequence=100.0,
    )
    fb = [
        FeedbackItem(
            component="Timing",
            severity="medium",
            message="Movement was slightly rushed.",
            recommendation="Pace yourself evenly.",
        )
    ]
    assessment = AssessmentResult(
        assessment_id="eval_001",
        attempt_session_id="sess_trainee",
        reference_profile_id="ref_001",
        activity_name="reach_and_pinch",
        overall_score=88.0,
        interpretation_band="Good match",
        component_scores=comp,
        critical_failures=[],
        feedback=fb,
    )
    temp_db.save_assessment(assessment)

    retrieved_eval = temp_db.get_assessment("eval_001")
    assert retrieved_eval is not None
    assert retrieved_eval.overall_score == 88.0
    assert retrieved_eval.interpretation_band == "Good match"
    assert retrieved_eval.component_scores.pose == 88.5
    assert len(retrieved_eval.feedback) == 1
    assert retrieved_eval.feedback[0].component == "Timing"


