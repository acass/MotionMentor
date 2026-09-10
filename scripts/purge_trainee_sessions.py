"""Delete every session of one role, what was derived from it, and its media.

Trainee sessions are practice attempts; expert sessions are the demonstrations
reference profiles are built from. Deleting a session also deletes any
reference profile built from it and any assessment scored from either, plus the
recording, landmark, feature and profile files on disk.

Run with no flags to review the manifest, then re-run with ``--apply``.
Use ``--role expert`` to clear expert captures instead of attempts, and
``--activity <name-or-id>`` to limit the purge to one activity.

The deletion logic lives in ``motion_mentor.purge`` so the dashboard's
"Clear Attempts" and "Clear Expert" buttons remove exactly the same things.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motion_mentor.purge import (  # noqa: E402
    apply_purge,
    backup_database,
    collect_files,
    collect_manifest,
)

DB_PATH = Path("data/motion_mentor.db")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the deletion")
    parser.add_argument(
        "--role",
        choices=("trainee", "expert"),
        default="trainee",
        help="which captures to clear (default: trainee)",
    )
    parser.add_argument("--activity", help="limit to one activity id or name")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"database not found: {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    try:
        manifest = collect_manifest(conn, args.role, args.activity)
        files = collect_files(manifest, DB_PATH.parent)
        print_manifest(manifest, files)

        if not any(manifest[k] for k in ("sessions", "references", "assessments")) and not files:
            print("\nnothing to purge.")
            return 0

        if not args.apply:
            print("\ndry run; nothing changed. re-run with --apply to delete.")
            return 0

        print(f"\nbacked up database to {backup_database(DB_PATH)}")
        counts = apply_purge(conn, manifest, files)
        print(
            f"deleted {counts['assessments']} assessment(s), {counts['references']} reference(s), "
            f"{counts['sessions']} session(s), {counts['files']} file(s)."
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
