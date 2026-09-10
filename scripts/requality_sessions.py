"""Re-run the browser-upload quality gate over stored sessions.

Sessions recorded before the browser-upload thresholds were relaxed are stored
with meets_criteria=False, which keeps expert takes out of every reference and
leaves trainee attempts unevaluated. This re-evaluates each session from its
saved landmark parquet, rebuilds any reference it unblocks, and scores trainee
attempts against it.

Usage: python scripts/requality_sessions.py [--apply]
"""

from __future__ import annotations

import argparse

from motion_mentor.server import (
    BROWSER_UPLOAD_QUALITY_EVALUATOR,
    get_mentor_app,
    process_completed_session,
)
from motion_mentor.storage.files import load_landmarks_parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Persist changes")
    args = parser.parse_args()

    mentor = get_mentor_app()
    for session in mentor.db.list_sessions():
        if not session.landmark_path:
            continue
        records = load_landmarks_parquet(session.landmark_path)
        summary = BROWSER_UPLOAD_QUALITY_EVALUATOR.evaluate(records=records)
        was = session.quality_summary.meets_criteria if session.quality_summary else None
        print(f"{session.session_id} {session.role}: {was} -> {summary.meets_criteria} {summary.status_reasons}")
        if not args.apply or summary.meets_criteria == was:
            continue
        session.quality_summary = summary
        mentor.db.save_session(session)

    if not args.apply:
        return

    # Experts first so their references exist before trainees are scored.
    sessions = sorted(mentor.db.list_sessions(), key=lambda s: s.role != "expert")
    for session in sessions:
        assessment, reference = process_completed_session(mentor, session)
        if reference:
            print(f"rebuilt reference for {session.activity_id}")
        if assessment:
            print(f"scored attempt {session.session_id}")


if __name__ == "__main__":
    main()
