"""Delete sessions of one role, everything derived from them, and their media.

Shared by the ``scripts/purge_trainee_sessions.py`` CLI and the
``/api/sessions/purge`` endpoint so both agree on what gets removed.

Deleting a session cascades: any reference profile built from it goes too, and
so does any assessment scored from the session or from a doomed profile.
Trainee sessions normally feed no reference profile, so clearing them touches
nothing else; clearing experts usually takes the references with them.

ponytail: raw SQL because DatabaseManager exposes no delete API; fold it in
there if deletion grows beyond this one case.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

Role = Literal["trainee", "expert"]

# Subdirectory of the data root -> filename patterns, formatted with the session id.
SESSION_FILE_PATTERNS = {
    "recordings": ["{id}.mp4"],
    "landmarks": ["{id}.parquet", "{id}_landmarks.parquet", "{id}_normalized.parquet"],
    "features": ["{id}_features.parquet", "{id}_plot.png"],
}
REFERENCE_FILE_PATTERNS = {
    "references": ["{id}_reference.parquet"],
}


def collect_manifest(
    conn: sqlite3.Connection, role: Role, activity: Optional[str] = None
) -> Dict[str, Any]:
    """Find the sessions of ``role``, plus the references and assessments built on them."""
    sql = "SELECT session_id FROM sessions WHERE role = ?"
    params: tuple = (role,)
    if activity:
        sql += (
            " AND (activity_id = ? OR activity_id IN"
            " (SELECT activity_id FROM activities WHERE name = ?))"
        )
        params += (activity, activity)
    doomed_sessions = {row[0] for row in conn.execute(sql, params)}

    doomed_references: set[str] = set()
    for ref_id, ids_json, medoid in conn.execute(
        "SELECT reference_id, expert_session_ids_json, medoid_session_id FROM reference_profiles"
    ):
        if (set(json.loads(ids_json)) | {medoid}) & doomed_sessions:
            doomed_references.add(ref_id)

    doomed_assessments: set[str] = set()
    for column, ids in (
        ("attempt_session_id", doomed_sessions),
        ("reference_profile_id", doomed_references),
    ):
        if not ids:
            continue
        placeholders = ",".join("?" * len(ids))
        doomed_assessments |= {
            row[0]
            for row in conn.execute(
                f"SELECT assessment_id FROM assessment_results WHERE {column} IN ({placeholders})",
                tuple(sorted(ids)),
            )
        }

    return {
        "role": role,
        "sessions": sorted(doomed_sessions),
        "references": sorted(doomed_references),
        "assessments": sorted(doomed_assessments),
    }


def collect_files(manifest: Dict[str, Any], data_root: Path) -> List[Path]:
    """Resolve the on-disk artifacts belonging to the doomed sessions and references.

    ``data_root`` is the directory holding the database, so a purge run against
    a copied database can never reach the live media files.
    """
    paths: List[Path] = []
    for ids, patterns in (
        (manifest["sessions"], SESSION_FILE_PATTERNS),
        (manifest["references"], REFERENCE_FILE_PATTERNS),
    ):
        for item_id in ids:
            for subdir, names in patterns.items():
                for name in names:
                    candidate = data_root / subdir / name.format(id=item_id)
                    if candidate.exists():
                        paths.append(candidate)
    return paths


def backup_database(db_path: Path) -> Path:
    """Copy the database next to itself with a UTC timestamp suffix."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_name(f"{db_path.name}.bak-{stamp}")
    shutil.copy2(db_path, backup)
    return backup


def apply_purge(
    conn: sqlite3.Connection, manifest: Dict[str, Any], files: List[Path]
) -> Dict[str, int]:
    """Delete rows in FK-safe order, then the media files."""
    with conn:  # one transaction; rolls back on exception
        for table, column, ids in (
            ("assessment_results", "assessment_id", manifest["assessments"]),
            ("reference_profiles", "reference_id", manifest["references"]),
            ("sessions", "session_id", manifest["sessions"]),
        ):
            if not ids:
                continue
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", tuple(ids))

    for path in files:
        path.unlink(missing_ok=True)

    return {
        "sessions": len(manifest["sessions"]),
        "references": len(manifest["references"]),
        "assessments": len(manifest["assessments"]),
        "files": len(files),
    }
