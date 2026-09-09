"""Session recorder orchestrating synchronized MP4 video and Parquet landmark persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple
import cv2
import numpy as np

from motion_mentor.storage.files import save_landmarks_parquet
from motion_mentor.storage.models import (
    Activity,
    HandLandmarkData,
    LandmarkFrameRecord,
    QualitySummary,
    Session,
    utc_now,
)

logger = logging.getLogger(__name__)


class SessionRecorder:
    """Records synchronized source video and landmark frames for an activity."""

    def __init__(
        self,
        session_id: str,
        activity: Activity,
        role: Literal["expert", "trainee"] = "expert",
        participant_id: str = "local-user",
        camera_id: str = "0",
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        video_output_path: str | Path = "data/recordings/session.mp4",
        landmark_output_path: str | Path = "data/landmarks/session.parquet",
    ) -> None:
        self.session_id = session_id
        self.activity = activity
        self.role = role
        self.participant_id = participant_id
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.video_output_path = Path(video_output_path)
        self.landmark_output_path = Path(landmark_output_path)

        # In-memory frame records
        self.frame_records: List[LandmarkFrameRecord] = []
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False
        self.start_time_iso = utc_now()
        self.first_timestamp_ms: Optional[float] = None
        self.last_timestamp_ms: Optional[float] = None
        self.frame_index = 0

    def start(self) -> None:
        """Initialize the video writer and prepare buffers."""
        self.video_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.landmark_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try browser-native H.264 (avc1) first, fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self.video_writer = cv2.VideoWriter(
            str(self.video_output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        if not self.video_writer.isOpened():
            # Fallback to alternate fourcc if avc1 is unsupported
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                str(self.video_output_path),
                fourcc,
                self.fps,
                (self.width, self.height),
            )

        if not self.video_writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter at {self.video_output_path}")

        self.start_time_iso = utc_now()
        self.is_recording = True
        self.frame_index = 0
        self.frame_records.clear()

    def record_frame(
        self,
        raw_frame: np.ndarray,
        timestamp_ms: float,
        hands: List[HandLandmarkData],
    ) -> None:
        """Write raw frame to video and append landmarks to record buffer."""
        if not self.is_recording or self.video_writer is None:
            return

        if self.first_timestamp_ms is None:
            self.first_timestamp_ms = timestamp_ms
        self.last_timestamp_ms = timestamp_ms

        # Write clean raw frame to video
        # Ensure frame dimensions match writer
        if raw_frame.shape[1] != self.width or raw_frame.shape[0] != self.height:
            resized = cv2.resize(raw_frame, (self.width, self.height))
            self.video_writer.write(resized)
        else:
            self.video_writer.write(raw_frame)

        # Record landmark frame
        record = LandmarkFrameRecord(
            session_id=self.session_id,
            frame_index=self.frame_index,
            timestamp_ms=timestamp_ms,
            hands=hands,
            valid=len(hands) > 0,
        )
        self.frame_records.append(record)
        self.frame_index += 1

    def finish(
        self,
        quality_summary: Optional[QualitySummary] = None,
    ) -> Tuple[Session, List[LandmarkFrameRecord]]:
        """Stop recording, finalize files, and return the completed Session model."""
        self.is_recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # Save Parquet landmarks
        save_landmarks_parquet(self.frame_records, self.landmark_output_path)

        # Ensure web-browser compatibility (H.264 + faststart) if ffmpeg is available
        if self.video_output_path.exists() and self.video_output_path.stat().st_size > 0:
            try:
                import shutil
                import subprocess
                if shutil.which("ffmpeg"):
                    temp_mp4 = self.video_output_path.with_suffix(".tmp.mp4")
                    cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", str(self.video_output_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(temp_mp4),
                    ]
                    res = subprocess.run(cmd, capture_output=True, timeout=15)
                    if res.returncode == 0 and temp_mp4.exists() and temp_mp4.stat().st_size > 0:
                        temp_mp4.replace(self.video_output_path)
            except Exception as e:
                logger.warning("Optional ffmpeg web-transcode skipped: %s", e)

        # Duration calculation
        duration_sec = 0.0
        if self.first_timestamp_ms is not None and self.last_timestamp_ms is not None:
            duration_sec = max(0.0, (self.last_timestamp_ms - self.first_timestamp_ms) / 1000.0)

        session = Session(
            session_id=self.session_id,
            activity_id=self.activity.activity_id,
            activity_name=self.activity.name,
            activity_version=self.activity.version,
            role=self.role,
            participant_id=self.participant_id,
            camera_id=self.camera_id,
            resolution=[self.width, self.height],
            nominal_fps=self.fps,
            model_version="hand_landmarker.task",
            started_at=self.start_time_iso,
            ended_at=utc_now(),
            duration_seconds=round(duration_sec, 2),
            total_frames=len(self.frame_records),
            video_path=str(self.video_output_path),
            landmark_path=str(self.landmark_output_path),
            quality_summary=quality_summary,
        )

        return session, self.frame_records
