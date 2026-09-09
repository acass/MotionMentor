"""File persistence layer for Parquet landmarks and JSON export/interchange."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from motion_mentor.storage.models import (
    HandLandmarkData,
    LandmarkFrameRecord,
    Session,
)


def save_landmarks_parquet(
    records: List[LandmarkFrameRecord],
    output_path: str | Path,
) -> None:
    """Save landmark frame records to a Parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in records:
        if not rec.hands:
            # Record frame even if no hands were detected (for gap/coverage analysis)
            rows.append({
                "session_id": rec.session_id,
                "frame_index": rec.frame_index,
                "timestamp_ms": rec.timestamp_ms,
                "hand_track_id": -1,
                "handedness": "None",
                "handedness_score": 0.0,
                "landmarks_image": [],
                "landmarks_world": [],
                "valid": False,
            })
        else:
            for hand in rec.hands:
                # Flatten 21x3 landmarks to 63 floats for max parquet compatibility
                img_flat = [float(coord) for pt in hand.landmarks_image for coord in pt]
                world_flat = [float(coord) for pt in hand.landmarks_world for coord in pt]
                rows.append({
                    "session_id": rec.session_id,
                    "frame_index": rec.frame_index,
                    "timestamp_ms": rec.timestamp_ms,
                    "hand_track_id": hand.hand_track_id,
                    "handedness": hand.handedness,
                    "handedness_score": float(hand.handedness_score),
                    "landmarks_image": img_flat,
                    "landmarks_world": world_flat,
                    "valid": bool(hand.valid),
                })

    df = pd.DataFrame(rows)
    # Define PyArrow schema with typed float lists
    schema = pa.schema([
        ("session_id", pa.string()),
        ("frame_index", pa.int64()),
        ("timestamp_ms", pa.float64()),
        ("hand_track_id", pa.int32()),
        ("handedness", pa.string()),
        ("handedness_score", pa.float32()),
        ("landmarks_image", pa.list_(pa.float32())),
        ("landmarks_world", pa.list_(pa.float32())),
        ("valid", pa.bool_()),
    ])

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")


def load_landmarks_parquet(file_path: str | Path) -> List[LandmarkFrameRecord]:
    """Load landmark records from a Parquet file back into LandmarkFrameRecord objects."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Landmarks file not found: {file_path}")

    table = pq.read_table(file_path)
    df = table.to_pandas()

    records_by_frame: Dict[int, LandmarkFrameRecord] = {}

    for _, row in df.iterrows():
        frame_idx = int(row["frame_index"])
        if frame_idx not in records_by_frame:
            records_by_frame[frame_idx] = LandmarkFrameRecord(
                session_id=str(row["session_id"]),
                frame_index=frame_idx,
                timestamp_ms=float(row["timestamp_ms"]),
                hands=[],
                valid=bool(row["valid"]),
            )

        if int(row["hand_track_id"]) >= 0:
            # Reconstruct 21x3 landmark arrays from 63 flat floats
            img_arr = np.array(row["landmarks_image"], dtype=np.float32).reshape(-1, 3).tolist()
            world_arr = np.array(row["landmarks_world"], dtype=np.float32).reshape(-1, 3).tolist()

            hand_data = HandLandmarkData(
                hand_track_id=int(row["hand_track_id"]),
                handedness=str(row["handedness"]),  # type: ignore
                handedness_score=float(row["handedness_score"]),
                landmarks_image=img_arr,
                landmarks_world=world_arr,
                valid=bool(row["valid"]),
            )
            records_by_frame[frame_idx].hands.append(hand_data)
            records_by_frame[frame_idx].valid = True

    # Return sorted by frame_index
    return [records_by_frame[idx] for idx in sorted(records_by_frame.keys())]


def export_session_json(
    session: Session,
    records: List[LandmarkFrameRecord],
    output_path: str | Path,
) -> None:
    """Export complete session metadata and landmarks as a standalone JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "session": session.model_dump(),
        "frames": [rec.model_dump() for rec in records],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_session_json(
    input_path: str | Path,
) -> tuple[Session, List[LandmarkFrameRecord]]:
    """Load session metadata and landmarks from a JSON interchange file."""
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session = Session.model_validate(data["session"])
    records = [LandmarkFrameRecord.model_validate(f) for f in data["frames"]]
    return session, records


def save_features_parquet(
    df_features: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save derived feature DataFrame to a Parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, compression="snappy", index=False)


def load_features_parquet(file_path: str | Path) -> pd.DataFrame:
    """Load derived features DataFrame from a Parquet file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Feature file not found: {file_path}")
    return pd.read_parquet(file_path)

