"""Unit tests for QualityEvaluator and reporting output."""

from motion_mentor.reporting.quality import QualityEvaluator
from motion_mentor.storage.models import (
    HandLandmarkData,
    LandmarkFrameRecord,
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


def test_browser_upload_evaluator_accepts_15_fps_webcam_take() -> None:
    """A clean 15 FPS browser capture must be able to back a reference profile."""
    from motion_mentor.server import BROWSER_UPLOAD_QUALITY_EVALUATOR

    records = [
        LandmarkFrameRecord(
            session_id="upload_sess",
            frame_index=i,
            timestamp_ms=float(i * (1000.0 / 15.0)),
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
        for i in range(75)
    ]

    summary = BROWSER_UPLOAD_QUALITY_EVALUATOR.evaluate(
        records, latencies_ms=[24.0] * 75, dropped_frames=0
    )

    assert summary.effective_fps == 15.0
    assert summary.meets_criteria is True, summary.status_reasons

    # A recording too coarse for motion analysis still fails.
    coarse = [
        LandmarkFrameRecord(
            session_id="upload_sess",
            frame_index=i,
            timestamp_ms=float(i * 200.0),
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
        for i in range(25)
    ]
    coarse_summary = BROWSER_UPLOAD_QUALITY_EVALUATOR.evaluate(
        coarse, latencies_ms=[24.0] * 25, dropped_frames=0
    )
    assert coarse_summary.meets_criteria is False
