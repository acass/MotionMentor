"""Remove fabricated (non-camera) captures and everything derived from them.

Sessions whose ``camera_id`` is in ``GENERATED_CAMERA_IDS`` were drawn or
injected rather than recorded. Any reference profile built on one is not a real
expert demonstration, and any assessment scored against such a profile is
meaningless. This deletes all three, plus the media on disk.

Run with ``--dry-run`` (the default) to review the manifest, then ``--apply``.

ponytail: raw SQL here because the storage layer exposes no delete API and this
is a one-shot cleanup. Move it into DatabaseManager only if deletion becomes a
product feature.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motion_mentor.server import GENERATED_CAMERA_IDS  # noqa: E402

DB_PATH = Path("data/motion_mentor.db")

# Directory -> filename patterns, formatted with the session id.
SESSION_FILE_PATTERNS = {
    Path("data/recordings"): ["{id}.mp4"],
    Path("data/landmarks"): ["{id}.parquet", "{id}_landmarks.parquet"],
    Path("data/features"): ["{id}_features.parquet"],
}
REFERENCE_FILE_PATTERNS = {
    Path("data/references"): ["{id}_reference.parquet"],
}


def collect_manifest(conn: sqlite3.Connection) -> dict:
    """Find every session, reference and assessment tainted by generated data."""
    placeholders = ",".join("?" * len(GENERATED_CAMERA_IDS))
    doomed_sessions = {
        row[0]
        for row in conn.execute(
            f"SELECT session_id FROM sessions WHERE camera_id IN ({placeholders})",
            tuple(sorted(GENERATED_CAMERA_IDS)),
        )
    }

    doomed_references: set[str] = set()
    surviving_references: dict[str, set[str]] = {}
    for ref_id, ids_json, medoid in conn.execute(
        "SELECT reference_id, expert_session_ids_json, medoid_session_id FROM reference_profiles"
    ):
        expert_ids = set(json.loads(ids_json)) | {medoid}
        if expert_ids & doomed_sessions:
            doomed_references.add(ref_id)
        else:
            surviving_references[ref_id] = expert_ids

    doomed_assessments: set[str] = set()
    if doomed_references:
        placeholders = ",".join("?" * len(doomed_references))
        doomed_assessments = {
            row[0]
            for row in conn.execute(
                "SELECT assessment_id FROM assessment_results "
                f"WHERE reference_profile_id IN ({placeholders})",
                tuple(sorted(doomed_references)),
            )
        }

    # A surviving reference must never end up pointing at a deleted session.
    dangling = {
        ref_id: sorted(ids & doomed_sessions)
        for ref_id, ids in surviving_references.items()
        if ids & doomed_sessions
    }

    return {
        "sessions": sorted(doomed_sessions),
        "references": sorted(doomed_references),
        "assessments": sorted(doomed_assessments),
        "dangling": dangling,
    }


def collect_files(manifest: dict) -> list[Path]:
    """Resolve on-disk artifacts belonging to the doomed sessions and references."""
    paths: list[Path] = []
    for ids, patterns in (
        (manifest["sessions"], SESSION_FILE_PATTERNS),
        (manifest["references"], REFERENCE_FILE_PATTERNS),
    ):
        for item_id in ids:
            for directory, names in patterns.items():
                for name in names:
                    candidate = directory / name.format(id=item_id)
                    if candidate.exists():
                        paths.append(candidate)
    return paths


def print_manifest(manifest: dict, files: list[Path]) -> None:
    for label in ("sessions", "references", "assessments"):
        rows = manifest[label]
        print(f"{label} to delete ({len(rows)}):")
        for row in rows:
            print(f"  {row}")
        if not rows:
            print("  (none)")
    print(f"files to delete ({len(files)}):")
    for path in files:
        print(f"  {path}")
    if not files:
        print("  (none)")


def apply_purge(conn: sqlite3.Connection, manifest: dict, files: list[Path]) -> None:
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
            print(f"deleted {len(ids)} row(s) from {table}")

    for path in files:
        path.unlink(missing_ok=True)
        print(f"deleted {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the deletion; without it the script only prints the manifest",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"database not found: {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    try:
        manifest = collect_manifest(conn)
        files = collect_files(manifest)
        print_manifest(manifest, files)

        if manifest["dangling"]:
            print("\nABORT: these surviving references still list a doomed session:", file=sys.stderr)
            for ref_id, ids in manifest["dangling"].items():
                print(f"  {ref_id} -> {', '.join(ids)}", file=sys.stderr)
            return 1

        if not any(manifest[k] for k in ("sessions", "references", "assessments")) and not files:
            print("\nnothing to purge.")
            return 0

        if not args.apply:
            print("\ndry run; nothing changed. re-run with --apply to delete.")
            return 0

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = DB_PATH.with_name(f"{DB_PATH.name}.bak-{stamp}")
        shutil.copy2(DB_PATH, backup)
        print(f"\nbacked up database to {backup}")

        apply_purge(conn, manifest, files)
        print("purge complete.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
