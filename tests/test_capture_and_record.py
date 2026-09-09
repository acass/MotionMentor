"""End-to-end integration tests for CameraCapture, SyntheticCamera, and SessionRecorder."""

from pathlib import Path
import cv2
from motion_mentor.capture.camera import SyntheticCamera
from motion_mentor.capture.recorder import SessionRecorder
from motion_mentor.storage.files import load_landmarks_parquet
from motion_mentor.storage.models import Activity, HandLandmarkData


def test_synthetic_camera_frame_generation() -> None:
    cam = SyntheticCamera(width=640, height=360, target_fps=30, total_seconds=1.0)
    ret, frame, ts_ms = cam.read()
    assert ret is True
    assert frame is not None
    assert frame.shape == (360, 640, 3)
    assert ts_ms >= 0.0
    cam.release()


def test_session_recorder_pipeline(tmp_path: Path) -> None:
    video_file = tmp_path / "test_session.mp4"
    landmarks_file = tmp_path / "test_session.parquet"

    act = Activity(activity_id="act_rec_01", name="reach_and_pinch")
    recorder = SessionRecorder(
        session_id="rec_sess_01",
        activity=act,
        role="expert",
        width=640,
        height=360,
        fps=30.0,
        video_output_path=video_file,
        landmark_output_path=landmarks_file,
    )

    recorder.start()
    cam = SyntheticCamera(width=640, height=360, target_fps=30, total_seconds=0.5)

    # Record 10 frames
    for i in range(10):
        ret, frame, ts_ms = cam.read()
        assert ret is True
        hand = HandLandmarkData(
            hand_track_id=0,
            handedness="Right",
            handedness_score=0.9,
            landmarks_image=[[0.5, 0.5, 0.0] for _ in range(21)],
            landmarks_world=[[0.0, 0.0, 0.0] for _ in range(21)],
            valid=True,
        )
        recorder.record_frame(frame, ts_ms, [hand])

    session, records = recorder.finish()

    assert video_file.exists()
    assert landmarks_file.exists()
    assert session.total_frames == 10
    assert len(records) == 10

    # Verify video can be opened by OpenCV
    cap = cv2.VideoCapture(str(video_file))
    assert cap.isOpened()
    read_ok, _ = cap.read()
    assert read_ok is True
    cap.release()

    # Verify parquet contents
    loaded_records = load_landmarks_parquet(landmarks_file)
    assert len(loaded_records) == 10
    assert loaded_records[0].hands[0].handedness == "Right"
