"""MotionMentor Application Service coordinating capture, tracking, storage, and replay."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple
import cv2
import numpy as np
import yaml

from motion_mentor.capture.camera import BaseCamera, CameraCapture, SyntheticCamera
from motion_mentor.capture.hud import HUDOverlay
from motion_mentor.capture.recorder import SessionRecorder
from motion_mentor.reporting.quality import (
    QualityEvaluator,
    print_terminal_quality_report,
)
from motion_mentor.storage.database import DatabaseManager
from motion_mentor.storage.files import (
    export_session_json,
    load_landmarks_parquet,
)
from motion_mentor.storage.models import (
    Activity,
    HandLandmarkData,
    LandmarkFrameRecord,
    QualitySummary,
    Session,
    generate_uuid,
)
from motion_mentor.tracking.hand_tracker import HandTracker

logger = logging.getLogger(__name__)


class MotionMentorApp:
    """Core controller for MotionMentor."""

    def __init__(self, config_path: str | Path = "configs/default.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        # Storage & Database
        db_path = self.config.get("storage", {}).get("database_path", "data/motion_mentor.db")
        self.db = DatabaseManager(db_path)

        # Quality evaluator
        self.quality_evaluator = QualityEvaluator()

        # HUD Overlay
        hud_cfg = self.config.get("hud", {})
        self.hud = HUDOverlay(
            show_skeleton=hud_cfg.get("show_skeleton", True),
            show_telemetry=hud_cfg.get("show_telemetry", True),
            show_capture_zone=hud_cfg.get("show_capture_zone", True),
            capture_zone_padding=hud_cfg.get("capture_zone_padding", 0.12),
        )

        # Lazy tracker
        self._tracker: Optional[HandTracker] = None

    def _load_config(self, path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def tracker(self) -> HandTracker:
        """Get or initialize the HandTracker instance."""
        if self._tracker is None:
            t_cfg = self.config.get("tracking", {})
            self._tracker = HandTracker(
                num_hands=t_cfg.get("num_hands", 2),
                min_detection_confidence=t_cfg.get("min_detection_confidence", 0.5),
                min_tracking_confidence=t_cfg.get("min_tracking_confidence", 0.5),
            )
        return self._tracker

    def load_or_create_activity(self, activity_path_or_name: str) -> Activity:
        """Load an activity from YAML config or database, or create default."""
        p = Path(activity_path_or_name)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            act = Activity.model_validate(data)
            self.db.save_activity(act)
            return act

        # Try by name in DB
        act = self.db.get_activity_by_name(activity_path_or_name)
        if act:
            return act

        # Check in configs/activities/
        p_act = Path(f"configs/activities/{activity_path_or_name}.yaml")
        if p_act.exists():
            with open(p_act, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            act = Activity.model_validate(data)
            self.db.save_activity(act)
            return act

        # Fallback default activity
        act = Activity(
            name=activity_path_or_name,
            version=1,
            description=f"Auto-generated activity for {activity_path_or_name}",
        )
        self.db.save_activity(act)
        return act

    def create_camera(
        self,
        camera_id: int | str = 0,
        use_synthetic: bool = False,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> BaseCamera:
        """Create physical or synthetic camera source."""
        if use_synthetic:
            return SyntheticCamera(width=width, height=height, target_fps=fps)
        try:
            return CameraCapture(
                device_id=camera_id,
                width=width,
                height=height,
                target_fps=fps,
            )
        except Exception as e:
            logger.warning("Failed to open camera %s: %s. Falling back to synthetic source.", camera_id, e)
            return SyntheticCamera(width=width, height=height, target_fps=fps)

    def close(self) -> None:
        """Cleanup resources."""
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
