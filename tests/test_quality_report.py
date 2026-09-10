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


def test_camera_drop_count_never_exceeds_frames_taken():
    """A caller slower than the camera skips stale frames instead of stalling,
    and the reported drop count stays bounded by the frames actually taken."""
    import threading
    import time

    import numpy as np

    from motion_mentor.capture.camera import CameraCapture

    class FakeCap:
        """Delivers a frame every 5 ms, like a camera running at 200 FPS."""

        def __init__(self):
            self.closed = False

        def read(self):
            time.sleep(0.005)
            if self.closed:
                return False, None
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def isOpened(self):
            return not self.closed

        def release(self):
            self.closed = True

    cam = CameraCapture.__new__(CameraCapture)
    cam.cap = FakeCap()
    cam.target_fps = 200
    cam.expected_frame_interval_ms = 5.0
    cam.frame_count = 0
    cam.dropped_frame_count = 0
    cam.last_timestamp_ms = 0.0
    cam.start_time_ms = 0.0
    cam._frame_ready = threading.Condition()
    cam._latest = None
    cam._captured = 0
    cam._consumed = 0
    cam._stopped = False
    cam._thread = threading.Thread(target=cam._pump, daemon=True)
    cam._thread.start()

    try:
        timestamps = []
        for _ in range(10):
            ret, frame, ts_ms = cam.read()
            assert ret and frame is not None
            timestamps.append(ts_ms)
            time.sleep(0.02)  # caller 4x slower than the camera
    finally:
        cam.release()

    assert cam.frame_count == 10
    assert len(set(timestamps)) == 10  # never handed the same frame twice
    assert cam.dropped_frame_count > 0  # slow caller really did skip frames
    assert cam.dropped_frame_count <= cam._captured
