"""FastAPI backend server for MotionMentor Web Dashboard and Playback UI."""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from motion_mentor.app import MotionMentorApp
from motion_mentor.purge import apply_purge, backup_database, collect_files, collect_manifest
from motion_mentor.reporting.quality import QualityEvaluator
from motion_mentor.storage.files import load_features_parquet, load_landmarks_parquet
from motion_mentor.storage.models import ReferenceProfile, Session

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MotionMentor API",
    description="Explainable Hand Motion Skill Assessment and Playback Service",
    version="1.0.0",
)

# Enable CORS for local development. No credentials are used, so a wildcard
# origin is safe here; keep credentials off so browsers do not treat the
# reflected origin as trusted.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """No favicon asset; answer quietly instead of a 404."""
    return Response(status_code=204)

_mentor_app: Optional[MotionMentorApp] = None

GENERATED_CAMERA_IDS = frozenset({"synthetic", "canonical-demonstration"})

# An uploaded browser recording is processed offline, frame by frame, so the
# "effective FPS" derived from its timestamps is just the webcam's capture rate
# re-stated - it says nothing about live pipeline throughput, which is what the
# 24 FPS live-capture gate exists to protect. Judging uploads by that gate fails
# every browser take, because MediaRecorder webcam capture runs at 15-30 FPS.
# The floor below still rejects a recording too coarse for motion analysis.
# ponytail: fixed floor; make it per-activity if some activity needs finer timing.
BROWSER_UPLOAD_MIN_FPS = 12.0
# Same reasoning for hand coverage: a browser take starts and ends with the hand
# entering and leaving frame, so a real 5-second take lands around 80-85%. The
# 90% gate exists for a controlled lab capture and fails every browser take,
# which leaves the expert with no reference and the trainee with no assessment.
# ponytail: fixed floor; make it per-activity if some activity needs stricter.
BROWSER_UPLOAD_MIN_COVERAGE_PCT = 70.0
BROWSER_UPLOAD_QUALITY_EVALUATOR = QualityEvaluator(
    min_hand_coverage_pct=BROWSER_UPLOAD_MIN_COVERAGE_PCT,
    min_median_fps=BROWSER_UPLOAD_MIN_FPS,
)


class CaptureKind(str, Enum):
    RECORDED = "recorded"
    GENERATED = "generated"
    MIXED = "mixed"
    UNAVAILABLE = "unavailable"


def get_mentor_app() -> MotionMentorApp:
    global _mentor_app
    if _mentor_app is None:
        _mentor_app = MotionMentorApp()
    return _mentor_app


def get_capture_kind(session: Session) -> CaptureKind:
    """Classify whether a stored session came from a real camera."""
    if session.camera_id in GENERATED_CAMERA_IDS:
        return CaptureKind.GENERATED
    return CaptureKind.RECORDED


def get_reference_representative(
    mentor: MotionMentorApp,
    profile: ReferenceProfile,
) -> Optional[Session]:
    """Load the session whose video and landmarks represent a reference."""
    return mentor.db.get_session(profile.medoid_session_id)


def get_reference_capture_kind(
    mentor: MotionMentorApp,
    profile: ReferenceProfile,
) -> CaptureKind:
    """Classify a reference from every expert take included in it."""
    kinds = set()
    for session_id in profile.expert_session_ids:
        session = mentor.db.get_session(session_id)
        if not session:
            return CaptureKind.UNAVAILABLE
        kinds.add(get_capture_kind(session))
    if len(kinds) != 1:
        return CaptureKind.MIXED
    return next(iter(kinds))


def reference_source_priority(kind: CaptureKind) -> int:
    """Prefer real recorded references over generated demonstrations."""
    return {
        CaptureKind.RECORDED: 0,
        CaptureKind.GENERATED: 1,
        CaptureKind.MIXED: 2,
        CaptureKind.UNAVAILABLE: 3,
    }[kind]


def list_selectable_references(
    mentor: MotionMentorApp,
    activity_id: Optional[str] = None,
) -> list[tuple[ReferenceProfile, CaptureKind]]:
    """Return homogeneous, available references with their source classification."""
    classified = []
    for profile in mentor.db.list_reference_profiles():
        if activity_id is not None and profile.activity_id != activity_id:
            continue
        kind = get_reference_capture_kind(mentor, profile)
        if kind in {CaptureKind.RECORDED, CaptureKind.GENERATED}:
            classified.append((profile, kind))
    classified.sort(key=lambda item: reference_source_priority(item[1]))
    return classified


def get_preferred_reference(
    mentor: MotionMentorApp,
    activity_id: str,
) -> Optional[ReferenceProfile]:
    """Return the newest recorded reference, falling back to generated data."""
    references = list_selectable_references(mentor, activity_id)
    if not references:
        return None
    return references[0][0]


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
    role: Optional[Literal["expert", "trainee"]] = Query(None),
) -> List[Dict[str, Any]]:
    """List recorded sessions with optional filters."""
    mentor = get_mentor_app()
    sessions = mentor.db.list_sessions(activity_id=activity_id, role=role)
    return [s.model_dump() for s in sessions]


@app.delete("/api/sessions/purge")
def purge_sessions(
    role: Literal["trainee", "expert"] = Query("trainee", description="Which captures to clear"),
    activity: Optional[str] = Query(None, description="Limit to one activity id or name"),
    apply: bool = Query(False, description="Actually delete; otherwise preview only"),
) -> Dict[str, Any]:
    """Preview or delete every session of one role, plus what was derived from it."""
    mentor = get_mentor_app()
    conn = mentor.db.get_connection()
    try:
        manifest = collect_manifest(conn, role, activity)
        files = collect_files(manifest, mentor.db.db_path.parent)
        preview = {
            "applied": False,
            "role": role,
            "sessions": manifest["sessions"],
            "references": manifest["references"],
            "assessments": manifest["assessments"],
            "files": [str(p) for p in files],
        }
        if not apply:
            return preview

        backup = backup_database(mentor.db.db_path)
        counts = apply_purge(conn, manifest, files)
        logger.info("purged %s sessions: %s (backup at %s)", role, counts, backup)
        return {**preview, "applied": True, "backup": str(backup), "deleted": counts}
    finally:
        conn.close()


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
    output: List[Dict[str, Any]] = []
    for profile, kind in list_selectable_references(mentor):
        item = profile.model_dump()
        representative = get_reference_representative(mentor, profile)
        if representative:
            item["representative_session"] = {
                "session_id": representative.session_id,
                "participant_id": representative.participant_id,
                "camera_id": representative.camera_id,
                "capture_kind": kind.value,
            }
        else:
            item["representative_session"] = None
        output.append(item)

    return output


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
def list_assessments(
    attempt_session_id: Optional[str] = Query(None),
    reference_id: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """List assessment results with optional attempt and reference filters."""
    mentor = get_mentor_app()
    assessments = mentor.db.list_assessments(
        attempt_session_id=attempt_session_id,
        reference_profile_id=reference_id,
    )
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
        preferred = get_preferred_reference(mentor, attempt.activity_id)
        if not preferred:
            raise HTTPException(status_code=400, detail="No reference profile found for this activity")
        ref_id = preferred.reference_id

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
    role: Literal["expert", "trainee"] = "trainee"
    participant_id: Optional[str] = "local-user"


class SyntheticRecordRequest(BaseModel):
    activity_id: Optional[str] = "reach-and-pinch-001"
    duration: Optional[float] = 5.0
    # Synthetic takes are drawn, not captured, so they may never stand in for an
    # expert demonstration. Rejected with a 422 by FastAPI.
    role: Literal["trainee"] = "trainee"
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

    summary = BROWSER_UPLOAD_QUALITY_EVALUATOR.evaluate(
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

    assessment_data, reference_data = process_completed_session(mentor, session)

    return {
        "session": session.model_dump(),
        "assessment": assessment_data,
        "reference": reference_data,
    }


def rebuild_reference_for_activity(
    mentor: MotionMentorApp,
    activity_id: str,
    anchor_session: Optional[Session] = None,
) -> Optional[Dict[str, Any]]:
    """Create a reference from compatible, quality-passing expert takes."""
    from motion_mentor.comparison.pipeline import prepare_session_features
    from motion_mentor.comparison.reference import ReferenceProfileBuilder

    activity = mentor.db.get_activity(activity_id)
    if not activity:
        logger.warning("Skipping reference rebuild: activity %s was not found", activity_id)
        return None

    expert_sessions = [
        session for session in mentor.db.list_sessions(activity_id=activity_id, role="expert")
        if session.quality_summary and session.quality_summary.meets_criteria
        # A reference is what the trainee is measured against, so only real
        # camera captures may back one. This is the single choke point every
        # reference build routes through.
        and get_capture_kind(session) is not CaptureKind.GENERATED
        and (
            anchor_session is None
            or get_capture_kind(session) == get_capture_kind(anchor_session)
        )
    ]
    if not expert_sessions:
        logger.warning("Skipping reference rebuild: no quality-passing expert takes for %s", activity_id)
        return None

    try:
        sessions_and_features = [
            (session, prepare_session_features(mentor, session))
            for session in expert_sessions
        ]
        profile, _ = ReferenceProfileBuilder(activity).build_profile(sessions_and_features)
        mentor.db.save_reference_profile(profile)
        return profile.model_dump()
    except Exception:
        logger.exception("Reference rebuild failed for activity %s", activity_id)
        return None


def process_completed_session(
    mentor: MotionMentorApp, session: Session
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Refresh the expert reference or assess a trainee session after recording."""
    if session.role == "expert":
        if not session.quality_summary or not session.quality_summary.meets_criteria:
            logger.warning(
                "Skipping reference rebuild: expert take %s did not meet quality criteria",
                session.session_id,
            )
            return None, None
        return None, rebuild_reference_for_activity(
            mentor,
            session.activity_id,
            anchor_session=session,
        )

    reference = get_preferred_reference(mentor, session.activity_id)
    if not reference:
        return None, None

    from motion_mentor.comparison.pipeline import compare_attempt_to_reference, prepare_session_features

    try:
        prepare_session_features(mentor, session)
        assessment = compare_attempt_to_reference(mentor, session, reference.reference_id)
        return assessment.model_dump(), None
    except Exception as e:
        logger.warning("Auto-evaluation failed: %s", e)
        return None, None


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
        role=req.role,
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

    assessment_data, reference_data = process_completed_session(mentor, session)

    return {
        "session": session.model_dump(),
        "assessment": assessment_data,
        "reference": reference_data,
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
