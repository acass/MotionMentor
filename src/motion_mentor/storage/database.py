"""SQLite database layer for MotionMentor."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from motion_mentor.storage.models import Activity, QualitySummary, Session


class DatabaseManager:
    """Manages SQLite storage for activities and sessions."""

    def __init__(self, db_path: str | Path = "data/motion_mentor.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_tables()

    def get_connection(self) -> sqlite3.Connection:
        """Create a sqlite connection with foreign keys and dict-like rows."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def init_tables(self) -> None:
        """Initialize database tables if they do not exist."""
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activities (
                    activity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    description TEXT,
                    expected_hands INTEGER NOT NULL DEFAULT 1,
                    allow_mirroring INTEGER NOT NULL DEFAULT 0,
                    target_fps INTEGER NOT NULL DEFAULT 30,
                    scoring_profile_id TEXT,
                    capture_instructions TEXT,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    activity_id TEXT NOT NULL,
                    activity_version INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    nominal_fps REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_seconds REAL DEFAULT 0.0,
                    total_frames INTEGER DEFAULT 0,
                    video_path TEXT,
                    landmark_path TEXT,
                    quality_summary_json TEXT,
                    FOREIGN KEY (activity_id) REFERENCES activities (activity_id)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_activity
                ON sessions (activity_id, role);
                """
            )

    def save_activity(self, activity: Activity) -> None:
        """Insert or replace an activity."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO activities (
                    activity_id, name, version, description,
                    expected_hands, allow_mirroring, target_fps,
                    scoring_profile_id, capture_instructions,
                    config_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    activity.activity_id,
                    activity.name,
                    activity.version,
                    activity.description,
                    activity.expected_hands,
                    1 if activity.allow_mirroring else 0,
                    activity.target_fps,
                    activity.scoring_profile_id,
                    activity.capture_instructions,
                    activity.model_dump_json(),
                    activity.created_at,
                ),
            )

    def get_activity(self, activity_id: str) -> Optional[Activity]:
        """Fetch an activity by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT config_json FROM activities WHERE activity_id = ?;",
                (activity_id,),
            )
            row = cursor.fetchone()
            if row:
                return Activity.model_validate_json(row["config_json"])
            return None

    def get_activity_by_name(self, name: str) -> Optional[Activity]:
        """Fetch an activity by name."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT config_json FROM activities WHERE name = ? ORDER BY version DESC LIMIT 1;",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                return Activity.model_validate_json(row["config_json"])
            return None

    def list_activities(self) -> List[Activity]:
        """List all activities."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT config_json FROM activities ORDER BY name ASC, version DESC;"
            )
            return [Activity.model_validate_json(r["config_json"]) for r in cursor.fetchall()]

    def save_session(self, session: Session) -> None:
        """Insert or update a session record."""
        # Ensure referenced activity exists in DB to prevent foreign key errors
        if not self.get_activity(session.activity_id):
            placeholder_act = Activity(
                activity_id=session.activity_id,
                name=session.activity_name or session.activity_id,
                version=session.activity_version,
            )
            self.save_activity(placeholder_act)

        quality_json = (
            session.quality_summary.model_dump_json()
            if session.quality_summary
            else None
        )
        res_str = f"{session.resolution[0]}x{session.resolution[1]}"

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    session_id, activity_id, activity_version, role,
                    participant_id, camera_id, resolution, nominal_fps,
                    model_version, started_at, ended_at, duration_seconds,
                    total_frames, video_path, landmark_path, quality_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    session.session_id,
                    session.activity_id,
                    session.activity_version,
                    session.role,
                    session.participant_id,
                    session.camera_id,
                    res_str,
                    session.nominal_fps,
                    session.model_version,
                    session.started_at,
                    session.ended_at,
                    session.duration_seconds,
                    session.total_frames,
                    session.video_path,
                    session.landmark_path,
                    quality_json,
                ),
            )

    def get_session(self, session_id: str) -> Optional[Session]:
        """Fetch a session by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.*, a.name as activity_name
                FROM sessions s
                LEFT JOIN activities a ON s.activity_id = a.activity_id
                WHERE s.session_id = ?;
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_session(row)

    def list_sessions(
        self, activity_id: Optional[str] = None, role: Optional[str] = None
    ) -> List[Session]:
        """List sessions with optional activity or role filters."""
        query = """
            SELECT s.*, a.name as activity_name
            FROM sessions s
            LEFT JOIN activities a ON s.activity_id = a.activity_id
        """
        params: List[str] = []
        conditions: List[str] = []
        if activity_id:
            conditions.append("s.activity_id = ?")
            params.append(activity_id)
        if role:
            conditions.append("s.role = ?")
            params.append(role)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY s.started_at DESC;"

        with self.get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            return [self._row_to_session(r) for r in cursor.fetchall()]

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """Convert a database row into a Session model."""
        res_parts = [int(x) for x in str(row["resolution"]).split("x")]
        quality = None
        if row["quality_summary_json"]:
            quality = QualitySummary.model_validate_json(row["quality_summary_json"])

        return Session(
            session_id=row["session_id"],
            activity_id=row["activity_id"],
            activity_name=row["activity_name"] or "",
            activity_version=row["activity_version"],
            role=row["role"],
            participant_id=row["participant_id"],
            camera_id=row["camera_id"],
            resolution=res_parts,
            nominal_fps=row["nominal_fps"],
            model_version=row["model_version"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            duration_seconds=row["duration_seconds"] or 0.0,
            total_frames=row["total_frames"] or 0,
            video_path=row["video_path"],
            landmark_path=row["landmark_path"],
            quality_summary=quality,
        )
