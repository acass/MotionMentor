"""Integration coverage for expert capture and reference selection workflows."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from motion_mentor.app import MotionMentorApp
import motion_mentor.server as server_module
from motion_mentor.server import process_completed_session
from motion_mentor.storage.files import save_features_parquet
from motion_mentor.storage.models import (
    Activity,
    AssessmentResult,
    QualitySummary,
    ReferenceProfile,
    Session,
)


@pytest.fixture
def workflow_context(tmp_path, monkeypatch) -> tuple[MotionMentorApp, Activity]:
    monkeypatch.chdir(tmp_path)
    mentor = MotionMentorApp(config_path=tmp_path / "missing-config.yaml")
    activity = Activity(activity_id="activity-1", name="reach_and_pinch")
    mentor.db.save_activity(activity)
    return mentor, activity


def make_quality(*, passes: bool) -> QualitySummary:
    coverage = 100.0 if passes else 70.0
    return QualitySummary(
        total_frames=2,
        frames_with_hand=2 if passes else 1,
        detection_coverage_pct=coverage,
        effective_fps=30.0,
        meets_criteria=passes,
        status_reasons=[] if passes else ["Hand detection coverage is too low."],
    )


def save_expert(
    mentor: MotionMentorApp,
    activity: Activity,
    *,
    session_id: str,
    camera_id: str,
    passes: bool,
    started_at: str,
) -> Session:
    session = Session(
        session_id=session_id,
        activity_id=activity.activity_id,
        activity_name=activity.name,
        role="expert",
        participant_id="expert",
        camera_id=camera_id,
        duration_seconds=1.0,
        total_frames=2,
        started_at=started_at,
        quality_summary=make_quality(passes=passes),
    )
    mentor.db.save_session(session)

    if passes:
        features = pd.DataFrame(
            {
                "frame_index": [0, 1],
                "timestamp_ms": [0.0, 33.3],
                "valid": [True, True],
                "index_mcp": [150.0, 145.0],
                "index_pip": [160.0, 155.0],
                "thumb_mcp": [140.0, 135.0],
                "thumb_ip": [150.0, 145.0],
                "pinch_distance": [0.8, 0.5],
                "palm_pitch": [10.0, 45.0],
                "palm_roll": [5.0, 30.0],
                "wrist_x": [0.4, 0.6],
                "wrist_y": [0.5, 0.4],
            }
        )
        save_features_parquet(
            features,
            Path(f"data/features/{session.session_id}_features.parquet"),
        )

    return session


def save_reference(
    mentor: MotionMentorApp,
    activity: Activity,
    expert: Session,
    *,
    reference_id: str,
    created_at: str,
) -> ReferenceProfile:
    features = pd.read_parquet(
        Path(f"data/features/{expert.session_id}_features.parquet")
    )
    envelope = {
        "frame_index": features["frame_index"],
        "timestamp_ms": features["timestamp_ms"],
    }
    for column in features.columns:
        if column in {"frame_index", "timestamp_ms", "valid"}:
            continue
        envelope[f"{column}_mean"] = features[column]
        envelope[f"{column}_std"] = 4.0

    profile_path = Path(f"data/references/{reference_id}.parquet")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(envelope).to_parquet(profile_path, index=False)
    profile = ReferenceProfile(
        reference_id=reference_id,
        activity_id=activity.activity_id,
        activity_name=activity.name,
        expert_session_ids=[expert.session_id],
        medoid_session_id=expert.session_id,
        total_demonstrations=1,
        duration_mean_sec=1.0,
        profile_path=str(profile_path),
        created_at=created_at,
    )
    mentor.db.save_reference_profile(profile)
    return profile


def test_failed_expert_capture_does_not_rebuild_from_older_takes(
    workflow_context,
) -> None:
    mentor, activity = workflow_context

    save_expert(
        mentor,
        activity,
        session_id="older-passing-expert",
        camera_id="canonical-demonstration",
        passes=True,
        started_at="2026-09-09T10:00:00+00:00",
    )
    failed_capture = save_expert(
        mentor,
        activity,
        session_id="new-failed-webcam-expert",
        camera_id="browser-webcam",
        passes=False,
        started_at="2026-09-09T11:00:00+00:00",
    )

    assessment, reference = process_completed_session(mentor, failed_capture)

    assert assessment is None
    assert reference is None
    assert mentor.db.list_reference_profiles() == []


def test_new_expert_reference_uses_only_compatible_capture_source(
    workflow_context,
) -> None:
    mentor, activity = workflow_context

    save_expert(
        mentor,
        activity,
        session_id="generated-expert",
        camera_id="canonical-demonstration",
        passes=True,
        started_at="2026-09-09T10:00:00+00:00",
    )
    save_expert(
        mentor,
        activity,
        session_id="local-camera-expert",
        camera_id="0",
        passes=True,
        started_at="2026-09-09T10:30:00+00:00",
    )
    webcam_capture = save_expert(
        mentor,
        activity,
        session_id="webcam-expert",
        camera_id="browser-webcam",
        passes=True,
        started_at="2026-09-09T11:00:00+00:00",
    )

    assessment, reference = process_completed_session(mentor, webcam_capture)

    assert assessment is None
    assert reference is not None
    assert reference["expert_session_ids"] == [
        "webcam-expert",
        "local-camera-expert",
    ]
    assert reference["medoid_session_id"] == "webcam-expert"


def test_reference_list_prefers_recorded_experts_and_identifies_representative(
    workflow_context, monkeypatch
) -> None:
    mentor, activity = workflow_context

    generated = save_expert(
        mentor,
        activity,
        session_id="generated-expert",
        camera_id="canonical-demonstration",
        passes=True,
        started_at="2026-09-09T12:00:00+00:00",
    )
    recorded = save_expert(
        mentor,
        activity,
        session_id="recorded-expert",
        camera_id="browser-webcam",
        passes=True,
        started_at="2026-09-09T11:00:00+00:00",
    )
    mentor.db.save_reference_profile(
        ReferenceProfile(
            reference_id="generated-reference",
            activity_id=activity.activity_id,
            activity_name=activity.name,
            expert_session_ids=[generated.session_id],
            medoid_session_id=generated.session_id,
            total_demonstrations=1,
            duration_mean_sec=1.0,
            profile_path="data/references/generated.parquet",
            created_at="2026-09-09T12:00:00+00:00",
        )
    )
    mentor.db.save_reference_profile(
        ReferenceProfile(
            reference_id="recorded-reference",
            activity_id=activity.activity_id,
            activity_name=activity.name,
            expert_session_ids=[recorded.session_id],
            medoid_session_id=recorded.session_id,
            total_demonstrations=1,
            duration_mean_sec=1.0,
            profile_path="data/references/recorded.parquet",
            created_at="2026-09-09T11:00:00+00:00",
        )
    )
    mentor.db.save_reference_profile(
        ReferenceProfile(
            reference_id="mixed-reference",
            activity_id=activity.activity_id,
            activity_name=activity.name,
            expert_session_ids=[recorded.session_id, generated.session_id],
            medoid_session_id=recorded.session_id,
            total_demonstrations=2,
            duration_mean_sec=1.0,
            profile_path="data/references/mixed.parquet",
            created_at="2026-09-09T13:00:00+00:00",
        )
    )
    monkeypatch.setattr(server_module, "_mentor_app", mentor)

    response = TestClient(server_module.app).get("/api/references")

    assert response.status_code == 200
    references = response.json()
    assert [item["reference_id"] for item in references] == [
        "recorded-reference",
        "generated-reference",
    ]
    assert references[0]["representative_session"] == {
        "session_id": "recorded-expert",
        "participant_id": "expert",
        "camera_id": "browser-webcam",
        "capture_kind": "recorded",
    }
    assert references[1]["representative_session"]["capture_kind"] == "generated"


def test_assessment_list_can_match_the_selected_reference(
    workflow_context, monkeypatch
) -> None:
    mentor, activity = workflow_context
    expert = save_expert(
        mentor,
        activity,
        session_id="recorded-expert",
        camera_id="browser-webcam",
        passes=True,
        started_at="2026-09-09T10:00:00+00:00",
    )
    attempt = Session(
        session_id="trainee-attempt",
        activity_id=activity.activity_id,
        activity_name=activity.name,
        role="trainee",
    )
    mentor.db.save_session(attempt)

    for reference_id in ("reference-a", "reference-b"):
        mentor.db.save_reference_profile(
            ReferenceProfile(
                reference_id=reference_id,
                activity_id=activity.activity_id,
                activity_name=activity.name,
                expert_session_ids=[expert.session_id],
                medoid_session_id=expert.session_id,
                total_demonstrations=1,
                duration_mean_sec=1.0,
                profile_path=f"data/references/{reference_id}.parquet",
            )
        )
        mentor.db.save_assessment(
            AssessmentResult(
                assessment_id=f"assessment-{reference_id}",
                attempt_session_id=attempt.session_id,
                reference_profile_id=reference_id,
                activity_name=activity.name,
                overall_score=80.0,
            )
        )
    monkeypatch.setattr(server_module, "_mentor_app", mentor)

    response = TestClient(server_module.app).get(
        "/api/assessments",
        params={
            "attempt_session_id": attempt.session_id,
            "reference_id": "reference-b",
        },
    )

    assert response.status_code == 200
    assessments = response.json()
    assert [item["reference_profile_id"] for item in assessments] == ["reference-b"]


def test_trainee_auto_comparison_prefers_recorded_reference(
    workflow_context,
) -> None:
    mentor, activity = workflow_context
    recorded_expert = save_expert(
        mentor,
        activity,
        session_id="recorded-expert",
        camera_id="browser-webcam",
        passes=True,
        started_at="2026-09-09T10:00:00+00:00",
    )
    generated_expert = save_expert(
        mentor,
        activity,
        session_id="generated-expert",
        camera_id="canonical-demonstration",
        passes=True,
        started_at="2026-09-09T11:00:00+00:00",
    )
    recorded_reference = save_reference(
        mentor,
        activity,
        recorded_expert,
        reference_id="recorded-reference",
        created_at="2026-09-09T10:00:00+00:00",
    )
    save_reference(
        mentor,
        activity,
        generated_expert,
        reference_id="generated-reference",
        created_at="2026-09-09T11:00:00+00:00",
    )

    attempt = Session(
        session_id="trainee-attempt",
        activity_id=activity.activity_id,
        activity_name=activity.name,
        role="trainee",
        duration_seconds=1.0,
        total_frames=2,
        quality_summary=make_quality(passes=True),
    )
    mentor.db.save_session(attempt)
    recorded_features = pd.read_parquet(
        Path(f"data/features/{recorded_expert.session_id}_features.parquet")
    )
    save_features_parquet(
        recorded_features,
        Path(f"data/features/{attempt.session_id}_features.parquet"),
    )

    assessment, reference = process_completed_session(mentor, attempt)

    assert reference is None
    assert assessment is not None
    assert assessment["reference_profile_id"] == recorded_reference.reference_id
