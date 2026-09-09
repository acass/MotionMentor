"""FastAPI backend server for MotionMentor Web Dashboard and Playback UI."""

from __future__ import annotations

import json
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
from motion_mentor.storage.models import AssessmentResult, ReferenceProfile, Session

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
        from scripts.compare_sessions import prepare_session_features
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

    from scripts.compare_sessions import compare_attempt_to_reference
    try:
        assessment = compare_attempt_to_reference(mentor, attempt, ref_id)
        return assessment.model_dump()
    except Exception as e:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")


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
