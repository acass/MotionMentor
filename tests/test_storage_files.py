"""Unit tests for Parquet and JSON landmark persistence."""

import json
from pathlib import Path
import numpy as np
from motion_mentor.storage.files import (
    export_session_json,
    load_landmarks_parquet,
    save_landmarks_parquet,
)
from motion_mentor.storage.models import (
    HandLandmarkData,
    LandmarkFrameRecord,
    Session,
)


def test_parquet_round_trip(tmp_path: Path) -> None:
    parquet_path = tmp_path / "landmarks.parquet"

    # Create synthetic frames: 2 frames with 1 hand, 1 frame with 0 hands (gap)
    frames = []
    for i in range(3):
        if i == 1:
            # Gap frame
            frames.append(
                LandmarkFrameRecord(
                    session_id="test_sess",
                    frame_index=i,
                    timestamp_ms=float(i * 33.3),
                    hands=[],
                    valid=False,
                )
            )
        else:
            # 21 landmarks
            img_pts = [[float(j * 0.04), float(j * 0.03), -0.01] for j in range(21)]
            world_pts = [[float(j * 0.005), float(j * 0.004), -0.002] for j in range(21)]
            hand = HandLandmarkData(
                hand_track_id=0,
                handedness="Right",
                handedness_score=0.95,
                landmarks_image=img_pts,
                landmarks_world=world_pts,
                valid=True,
            )
            frames.append(
                LandmarkFrameRecord(
                    session_id="test_sess",
                    frame_index=i,
                    timestamp_ms=float(i * 33.3),
                    hands=[hand],
                    valid=True,
                )
            )

    save_landmarks_parquet(frames, parquet_path)
    assert parquet_path.exists()

    loaded = load_landmarks_parquet(parquet_path)
    assert len(loaded) == 3
    assert loaded[0].frame_index == 0
    assert len(loaded[0].hands) == 1
    assert loaded[0].hands[0].handedness == "Right"
    assert len(loaded[0].hands[0].landmarks_image) == 21
    # Check numerical precision
    np.testing.assert_allclose(
        loaded[0].hands[0].landmarks_image[5],
        frames[0].hands[0].landmarks_image[5],
        rtol=1e-5,
    )

    # Check gap frame
    assert loaded[1].frame_index == 1
    assert len(loaded[1].hands) == 0


def test_json_export_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "export.json"
    session = Session(
        session_id="sess_json_01",
        activity_id="act_01",
        role="trainee",
        duration_seconds=2.5,
        total_frames=75,
    )
    records = [
        LandmarkFrameRecord(
            session_id="sess_json_01",
            frame_index=0,
            timestamp_ms=0.0,
            hands=[],
            valid=False,
        )
    ]

    export_session_json(session, records, json_path)
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    assert data["session"]["session_id"] == "sess_json_01"
    assert data["session"]["role"] == "trainee"
    assert len(data["frames"]) == 1
