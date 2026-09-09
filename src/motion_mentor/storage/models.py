"""Pydantic data models for MotionMentor storage and schema validation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


def generate_uuid() -> str:
    """Generate a UUID4 string."""
    return str(uuid4())


def utc_now() -> str:
    """Return ISO-8601 formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class Activity(BaseModel):
    """Activity definition defining expectations and scoring parameters."""
    activity_id: str = Field(default_factory=generate_uuid)
    name: str
    version: int = 1
    description: str = ""
    expected_hands: int = 1
    allow_mirroring: bool = False
    target_fps: int = 30
    scoring_profile_id: Optional[str] = None
    capture_instructions: str = "Keep the full hand visible inside the frame."
    scoring_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "pose": 0.25,
            "trajectory": 0.25,
            "orientation": 0.15,
            "timing": 0.15,
            "smoothness": 0.10,
            "sequence": 0.10,
        }
    )
    checkpoints: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class QualitySummary(BaseModel):
    """Quality metrics summarizing capture and tracking fidelity."""
    total_frames: int = 0
    frames_with_hand: int = 0
    detection_coverage_pct: float = 0.0
    dropped_frames: int = 0
    effective_fps: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    jitter_std_ms: float = 0.0
    meets_criteria: bool = False
    status_reasons: List[str] = Field(default_factory=list)


class Session(BaseModel):
    """Session record representing an expert or trainee recording attempt."""
    session_id: str = Field(default_factory=generate_uuid)
    activity_id: str
    activity_name: str = ""
    activity_version: int = 1
    role: Literal["expert", "trainee"] = "expert"
    participant_id: str = "local-user"
    camera_id: str = "0"
    resolution: List[int] = Field(default_factory=lambda: [1280, 720])
    nominal_fps: float = 30.0
    model_version: str = "hand-landmarker-task-v1"
    started_at: str = Field(default_factory=utc_now)
    ended_at: Optional[str] = None
    duration_seconds: float = 0.0
    total_frames: int = 0
    video_path: Optional[str] = None
    landmark_path: Optional[str] = None
    quality_summary: Optional[QualitySummary] = None


class LandmarkPoint(BaseModel):
    """A single 3D landmark point."""
    x: float
    y: float
    z: float


class HandLandmarkData(BaseModel):
    """Landmark data for a single detected hand in a frame."""
    hand_track_id: int = 0
    handedness: Literal["Right", "Left", "Unknown"] = "Unknown"
    handedness_score: float = 0.0
    # 21 ordered landmarks
    landmarks_image: List[List[float]] = Field(
        default_factory=list,
        description="21 normalized [x, y, z] image coordinates in range [0, 1]."
    )
    landmarks_world: List[List[float]] = Field(
        default_factory=list,
        description="21 metric 3D [x, y, z] world coordinates in meters."
    )
    valid: bool = True


class LandmarkFrameRecord(BaseModel):
    """Full frame record containing timestamp and detected hands."""
    session_id: str
    frame_index: int
    timestamp_ms: float
    hands: List[HandLandmarkData] = Field(default_factory=list)
    valid: bool = True


class ComponentScores(BaseModel):
    """Component scores breakdown (0-100)."""
    pose: float = 0.0
    trajectory: float = 0.0
    orientation: float = 0.0
    timing: float = 0.0
    smoothness: float = 0.0
    sequence: float = 0.0


class FeedbackItem(BaseModel):
    """Actionable coaching instruction linked to measurable deviation and timestamp."""
    component: str
    severity: Literal["high", "medium", "low"] = "medium"
    message: str
    time_sec: float = 0.0
    measured_deviation: str = ""
    recommendation: str = ""


class AssessmentResult(BaseModel):
    """Final assessment result comparing a trainee attempt with an expert reference."""
    assessment_id: str = Field(default_factory=generate_uuid)
    attempt_session_id: str
    reference_profile_id: str
    activity_name: str = ""
    scoring_version: str = "1.0.0"
    overall_score: float = 0.0
    reliability: Literal["high", "medium", "low"] = "high"
    interpretation_band: str = "Developing"
    component_scores: ComponentScores = Field(default_factory=ComponentScores)
    critical_failures: List[str] = Field(default_factory=list)
    feedback: List[FeedbackItem] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class ReferenceProfile(BaseModel):
    """Expert reference profile metadata combining multiple demonstrations."""
    reference_id: str = Field(default_factory=generate_uuid)
    activity_id: str
    activity_name: str = ""
    activity_version: int = 1
    version: int = 1
    expert_session_ids: List[str] = Field(default_factory=list)
    medoid_session_id: str = ""
    total_demonstrations: int = 0
    duration_mean_sec: float = 0.0
    profile_path: str = ""
    created_at: str = Field(default_factory=utc_now)

