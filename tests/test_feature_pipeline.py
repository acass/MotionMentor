"""End-to-end integration test for the full feature processing pipeline."""

from pathlib import Path
import numpy as np
from motion_mentor.processing.features import extract_session_features_df
from motion_mentor.processing.normalization import normalize_session_records
from motion_mentor.processing.smoothing import smooth_landmark_records
from motion_mentor.storage.files import (
    load_features_parquet,
    save_features_parquet,
    save_landmarks_parquet,
)
from motion_mentor.storage.models import HandLandmarkData, LandmarkFrameRecord


def test_full_processing_pipeline(tmp_path: Path) -> None:
    raw_file = tmp_path / "raw_landmarks.parquet"
    norm_file = tmp_path / "normalized_landmarks.parquet"
    feat_file = tmp_path / "features.parquet"

    # Create 30 synthetic frames
    records = []
    for i in range(30):
        # Create realistic hand layout
        lms = [[0.0, 0.0, 0.0] for _ in range(21)]
        lms[0] = [0.5, 0.7, 0.0]  # Wrist
        lms[9] = [0.5, 0.5, 0.0]  # Middle MCP
        lms[4] = [0.45, 0.45, 0.0] # Thumb tip
        lms[8] = [0.48, 0.45, 0.0] # Index tip
        hand = HandLandmarkData(
            hand_track_id=0,
            handedness="Right",
            handedness_score=0.98,
            landmarks_image=lms,
            landmarks_world=lms,
            valid=True,
        )
        records.append(
            LandmarkFrameRecord(
                session_id="pipeline_test",
                frame_index=i,
                timestamp_ms=float(i * 33.33),
                hands=[hand],
                valid=True,
            )
        )

    save_landmarks_parquet(records, raw_file)

    # 1. Smooth
    smoothed = smooth_landmark_records(records)
    assert len(smoothed) == 30

    # 2. Normalize
    normalized, global_wrist, global_orient = normalize_session_records(smoothed)
    assert len(normalized) == 30
    save_landmarks_parquet(normalized, norm_file)
    assert norm_file.exists()

    # 3. Extract Features
    df_features = extract_session_features_df(
        normalized, global_trajectory=global_wrist, global_orientations=global_orient
    )
    assert not df_features.empty
    assert len(df_features) == 30

    # 4. Save and Load Features Parquet
    save_features_parquet(df_features, feat_file)
    assert feat_file.exists()

    loaded_df = load_features_parquet(feat_file)
    assert loaded_df.shape == df_features.shape
    assert "thumb_mcp" in loaded_df.columns
    assert "index_pip" in loaded_df.columns
    assert "pinch_distance" in loaded_df.columns
    assert "wrist_velocity" in loaded_df.columns
