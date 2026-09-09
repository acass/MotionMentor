"""FastAPI backend server for MotionMentor Web Dashboard and Playback UI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from motion_mentor.app import MotionMentorApp
from motion_mentor.storage.files import load_features_parquet, load_landmarks_parquet
from motion_mentor.storage.models import Session

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MotionMentor API",
    description="Explainable Hand Motion Skill Assessment and Playback Service",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_mentor_app: Optional[MotionMentorApp] = None


def get_mentor_app() -> MotionMentorApp:
    global _mentor_app
    if _mentor_app is None:
        _mentor_app = MotionMentorApp()
    return _mentor_app


# Request / Response Schemas
class CompareRequest(BaseModel):
    attempt_session_id: str
    reference_id: Optional[str] = None


# API Endpoints
@app.get("/api/activities")
def list_activities() -> List[Dict[str, Any]]:
    """List all registered activities."""
    mentor = get_mentor_app()
    activities = mentor.db.list_activities()
    return [a.model_dump() for a in activities]


@app.get("/api/sessions")
def list_sessions(
    activity_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """List recorded sessions with optional filters."""
    mentor = get_mentor_app()
    sessions = mentor.db.list_sessions(activity_id=activity_id, role=role)
    return [s.model_dump() for s in sessions]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    """Get metadata for a specific session."""
    mentor = get_mentor_app()
    session = mentor.db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return session.model_dump()


@app.get("/api/sessions/{session_id}/landmarks")
def get_session_landmarks(session_id: str) -> List[Dict[str, Any]]:
    """Get frame-by-frame 3D hand landmarks for playback."""
    mentor = get_mentor_app()
    session = mentor.db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    if not session.landmark_path or not Path(session.landmark_path).exists():
        raise HTTPException(status_code=404, detail="Landmark file missing for session")

    records = load_landmarks_parquet(session.landmark_path)
    output = []
    for r in records:
        hands_data = []
        for h in r.hands:
            hands_data.append({
                "hand_track_id": h.hand_track_id,
                "handedness": h.handedness,
                "landmarks": h.landmarks_image,
                "landmarks_world": h.landmarks_world,
            })
        output.append({
            "frame_index": r.frame_index,
            "timestamp_ms": r.timestamp_ms,
            "valid": r.valid,
            "hands": hands_data,
        })
    return output


@app.get("/api/sessions/{session_id}/features")
def get_session_features(session_id: str) -> Dict[str, Any]:
    """Get feature time-series (joint angles, velocities, etc.)."""
    feat_path = Path(f"data/features/{session_id}_features.parquet")
    if not feat_path.exists():
        # Try preparing features
        mentor = get_mentor_app()
        session = mentor.db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        from motion_mentor.comparison.pipeline import prepare_session_features
        try:
            df = prepare_session_features(mentor, session)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to prepare features: {e}")
    else:
        df = load_features_parquet(feat_path)

    return {
        "columns": list(df.columns),
        "records": df.to_dict(orient="records"),
    }


@app.get("/api/references")
def list_references() -> List[Dict[str, Any]]:
    """List all expert reference profiles."""
    mentor = get_mentor_app()
    profiles = mentor.db.list_reference_profiles()
    return [p.model_dump() for p in profiles]


@app.get("/api/references/{reference_id}")
def get_reference(reference_id: str) -> Dict[str, Any]:
    """Get reference profile metadata and mean/tolerance envelope."""
    mentor = get_mentor_app()
    profile = mentor.db.get_reference_profile(reference_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Reference profile not found: {reference_id}")

    envelope_records = []
    if profile.profile_path and Path(profile.profile_path).exists():
        df = pd.read_parquet(profile.profile_path)
        envelope_records = df.to_dict(orient="records")

    return {
        "profile": profile.model_dump(),
        "envelope": envelope_records,
    }


@app.get("/api/assessments")
def list_assessments(attempt_session_id: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
    """List assessment results with optional attempt session filter."""
    mentor = get_mentor_app()
    assessments = mentor.db.list_assessments(attempt_session_id=attempt_session_id)
    return [a.model_dump() for a in assessments]


@app.get("/api/assessments/{assessment_id}")
def get_assessment(assessment_id: str) -> Dict[str, Any]:
    """Get a specific assessment result."""
    mentor = get_mentor_app()
    assessment = mentor.db.get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail=f"Assessment not found: {assessment_id}")
    return assessment.model_dump()


@app.post("/api/compare")
def compare_attempt(req: CompareRequest) -> Dict[str, Any]:
    """Trigger comparison of a trainee session against a reference profile."""
    mentor = get_mentor_app()
    attempt = mentor.db.get_session(req.attempt_session_id)
    if not attempt:
        raise HTTPException(status_code=404, detail=f"Attempt session not found: {req.attempt_session_id}")

    ref_id = req.reference_id
    if not ref_id:
        latest = mentor.db.get_latest_reference_profile(attempt.activity_id)
        if not latest:
            raise HTTPException(status_code=400, detail="No reference profile found for this activity")
        ref_id = latest.reference_id

    from motion_mentor.comparison.pipeline import compare_attempt_to_reference
    try:
        assessment = compare_attempt_to_reference(mentor, attempt, ref_id)
        return assessment.model_dump()
    except Exception as e:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")


class RecordUploadRequest(BaseModel):
    video_base64: str
    mime_type: Optional[str] = "video/webm"
    activity_id: Optional[str] = "reach-and-pinch-001"
    role: Optional[str] = "trainee"
    participant_id: Optional[str] = "local-user"


class SyntheticRecordRequest(BaseModel):
    activity_id: Optional[str] = "reach-and-pinch-001"
    duration: Optional[float] = 5.0
    role: Optional[str] = "trainee"
    participant_id: Optional[str] = "local-user"


@app.post("/api/record_upload")
def record_upload(req: RecordUploadRequest) -> Dict[str, Any]:
    """Process a video recorded directly in the web browser."""
    import base64
    import shutil
    import subprocess
    import tempfile
    import cv2
    from motion_mentor.storage.models import generate_uuid, LandmarkFrameRecord, utc_now
    from motion_mentor.storage.files import save_landmarks_parquet

    mentor = get_mentor_app()
    session_id = generate_uuid()

    # Decode base64 payload
    try:
        # Strip data URL prefix if present
        data_str = req.video_base64
        if "," in data_str:
            data_str = data_str.split(",", 1)[1]
        video_bytes = base64.b64decode(data_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 video payload: {e}")

    temp_dir = Path(tempfile.gettempdir())
    raw_tmp = temp_dir / f"{session_id}_upload.raw"
    raw_tmp.write_bytes(video_bytes)

    rec_dir = Path("data/recordings")
    rec_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = rec_dir / f"{session_id}.mp4"

    # Convert to browser H.264 MP4 with ffmpeg if available
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(raw_tmp),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(mp4_path),
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            logger.warning("ffmpeg conversion failed: %s, falling back", res.stderr)
            shutil.copyfile(str(raw_tmp), str(mp4_path))
        raw_tmp.unlink(missing_ok=True)
    else:
        shutil.move(str(raw_tmp), str(mp4_path))

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not decode video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    records: List[LandmarkFrameRecord] = []
    latencies: List[float] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        ts_ms = frame_idx * (1000.0 / fps)
        hands, latency_ms = mentor.tracker.process_frame(frame, ts_ms)
        latencies.append(latency_ms)
        records.append(
            LandmarkFrameRecord(
                session_id=session_id,
                frame_index=frame_idx,
                timestamp_ms=ts_ms,
                hands=hands,
                valid=len(hands) > 0,
            )
        )
        frame_idx += 1
    cap.release()

    if not records:
        raise HTTPException(status_code=400, detail="Video contained 0 readable frames")

    # Save landmarks
    lm_dir = Path("data/landmarks")
    lm_dir.mkdir(parents=True, exist_ok=True)
    lm_path = lm_dir / f"{session_id}.parquet"
    save_landmarks_parquet(records, lm_path)

    summary = mentor.quality_evaluator.evaluate(
        records=records,
        latencies_ms=latencies,
        dropped_frames=0,
    )

    activity = mentor.db.get_activity(req.activity_id or "reach-and-pinch-001")
    act_name = activity.name if activity else "reach_and_pinch"
    act_version = activity.version if activity else 1
    duration_sec = round(len(records) / fps, 2)

    session = Session(
        session_id=session_id,
        activity_id=req.activity_id or "reach-and-pinch-001",
        activity_name=act_name,
        activity_version=act_version,
        role=req.role or "trainee",
        participant_id=req.participant_id or "local-user",
        camera_id="browser-webcam",
        resolution=[width, height],
        nominal_fps=round(fps, 1),
        model_version="hand_landmarker.task",
        started_at=utc_now(),
        ended_at=utc_now(),
        duration_seconds=duration_sec,
        total_frames=len(records),
        video_path=str(mp4_path),
        landmark_path=str(lm_path),
        quality_summary=summary,
    )
    mentor.db.save_session(session)

    # Auto-evaluate against reference
    assessment_data = None
    ref = mentor.db.get_latest_reference_profile(session.activity_id)
    if ref:
        from motion_mentor.comparison.pipeline import prepare_session_features, compare_attempt_to_reference
        try:
            prepare_session_features(mentor, session)
            assessment = compare_attempt_to_reference(mentor, session, ref.reference_id)
            assessment_data = assessment.model_dump()
        except Exception as e:
            logger.warning("Auto-evaluation failed: %s", e)

    return {
        "session": session.model_dump(),
        "assessment": assessment_data,
    }


@app.post("/api/record_synthetic")
def record_synthetic(req: SyntheticRecordRequest) -> Dict[str, Any]:
    """Generate a quick synthetic trainee attempt for testing without webcam hardware."""
    from motion_mentor.capture.camera import SyntheticCamera
    from motion_mentor.capture.recorder import SessionRecorder
    from motion_mentor.storage.models import generate_uuid

    mentor = get_mentor_app()
    activity = mentor.load_or_create_activity("reach_and_pinch")
    session_id = generate_uuid()

    video_out = Path(f"data/recordings/{session_id}.mp4")
    landmarks_out = Path(f"data/landmarks/{session_id}.parquet")

    duration = req.duration or 5.0
    camera = SyntheticCamera(width=1280, height=720, target_fps=30, total_seconds=duration)
    recorder = SessionRecorder(
        session_id=session_id,
        activity=activity,
        role=req.role or "trainee",
        participant_id=req.participant_id or "synthetic-user",
        camera_id="synthetic",
        width=1280,
        height=720,
        fps=30.0,
        video_output_path=video_out,
        landmark_output_path=landmarks_out,
    )

    recorder.start()
    latencies = []
    while True:
        ret, frame, ts_ms = camera.read()
        if not ret or frame is None:
            break
        hands, latency_ms = mentor.tracker.process_frame(frame, ts_ms)
        latencies.append(latency_ms)
        recorder.record_frame(frame, ts_ms, hands)

    summary = mentor.quality_evaluator.evaluate(
        records=recorder.frame_records,
        latencies_ms=latencies,
        dropped_frames=0,
    )
    session, _ = recorder.finish(quality_summary=summary)
    mentor.db.save_session(session)

    assessment_data = None
    ref = mentor.db.get_latest_reference_profile(activity.activity_id)
    if ref:
        from motion_mentor.comparison.pipeline import prepare_session_features, compare_attempt_to_reference
        try:
            prepare_session_features(mentor, session)
            assessment = compare_attempt_to_reference(mentor, session, ref.reference_id)
            assessment_data = assessment.model_dump()
        except Exception as e:
            logger.warning("Auto-evaluation failed: %s", e)

    return {
        "session": session.model_dump(),
        "assessment": assessment_data,
    }


@app.get("/api/videos/{video_filename}")
def stream_video(video_filename: str) -> FileResponse:
    """Stream recorded MP4 video with HTTP 206 Partial Content range support."""
    # Prevent directory traversal
    clean_name = os.path.basename(video_filename)
    path = Path("data/recordings") / clean_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {clean_name}")

    return FileResponse(
        path=path,
        media_type="video/mp4",
        filename=clean_name,
    )


# Mount static files for the web interface
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch uvicorn server."""
    import uvicorn
    uvicorn.run("motion_mentor.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
