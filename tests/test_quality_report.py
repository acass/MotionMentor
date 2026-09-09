"""Unit tests for QualityEvaluator and reporting output."""

from motion_mentor.reporting.quality import (
    QualityEvaluator,
    generate_markdown_quality_report,
)
from motion_mentor.storage.models import (
    HandLandmarkData,
    LandmarkFrameRecord,
    Session,
)


def test_quality_evaluator_pass() -> None:
    evaluator = QualityEvaluator(min_hand_coverage_pct=90.0, min_median_fps=24.0)

    records = []
    # 30 frames at 30 FPS with hand detected
    for i in range(30):
        records.append(
            LandmarkFrameRecord(
                session_id="eval_sess",
                frame_index=i,
                timestamp_ms=float(i * 33.33),
                hands=[
                    HandLandmarkData(
                        hand_track_id=0,
                        handedness="Right",
                        handedness_score=0.95,
                        valid=True,
                    )
                ],
                valid=True,
            )
        )

    latencies = [15.0] * 30
    summary = evaluator.evaluate(records, latencies_ms=latencies, dropped_frames=0)

    assert summary.total_frames == 30
    assert summary.frames_with_hand == 30
    assert summary.detection_coverage_pct == 100.0
    assert summary.effective_fps >= 29.0
    assert summary.median_latency_ms == 15.0
    assert summary.meets_criteria is True
    assert len(summary.status_reasons) == 0


def test_quality_evaluator_fail_coverage() -> None:
    evaluator = QualityEvaluator(min_hand_coverage_pct=90.0)

    records = []
    for i in range(10):
        # Only 5 frames have hands
        hands = (
            [
                HandLandmarkData(
                    hand_track_id=0,
                    handedness="Right",
                    handedness_score=0.9,
                    valid=True,
                )
            ]
            if i < 5
            else []
        )
        records.append(
            LandmarkFrameRecord(
                session_id="eval_sess",
                frame_index=i,
                timestamp_ms=float(i * 33.33),
                hands=hands,
                valid=len(hands) > 0,
            )
        )

    summary = evaluator.evaluate(records, latencies_ms=[20.0] * 10, dropped_frames=0)
    assert summary.detection_coverage_pct == 50.0
    assert summary.meets_criteria is False
    assert any("Hand detection coverage" in s for s in summary.status_reasons)


def test_markdown_report_generation() -> None:
    evaluator = QualityEvaluator()
    records = [
        LandmarkFrameRecord(
            session_id="md_sess",
            frame_index=0,
            timestamp_ms=0.0,
            hands=[],
            valid=False,
        )
    ]
    summary = evaluator.evaluate(records)
    session = Session(
        session_id="md_sess",
        activity_id="reach_and_pinch",
        role="expert",
        duration_seconds=1.0,
        total_frames=1,
    )
    md = generate_markdown_quality_report(session, summary)
    assert "# Capture Quality Report: md_sess" in md
    assert "reach_and_pinch" in md
