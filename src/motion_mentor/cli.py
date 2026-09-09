"""Unified Command Line Interface for MotionMentor."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
from rich.console import Console
from rich.table import Table

from motion_mentor.app import MotionMentorApp
from motion_mentor.capture.camera import enumerate_cameras
from motion_mentor.capture.recorder import SessionRecorder
from motion_mentor.reporting.quality import print_terminal_quality_report
from motion_mentor.storage.files import export_session_json, load_landmarks_parquet
from motion_mentor.storage.models import generate_uuid

console = Console()


def cmd_test_camera(args: argparse.Namespace) -> None:
    """Run camera validation and skeleton overlay test."""
    app = MotionMentorApp(args.config)
    console.print("[bold cyan]MotionMentor Camera Test & Feasibility Validator[/bold cyan]")

    # Check available cameras if physical
    if not args.synthetic:
        devices = enumerate_cameras()
        console.print(f"Detected camera device indices: {devices or 'None found'}")

    camera = app.create_camera(
        camera_id=args.camera_id,
        use_synthetic=args.synthetic,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    records = []
    latencies = []
    window_name = "MotionMentor - Camera Test (Press 'q' or ESC to exit)"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    t_start = time.time()
    rolling_fps = 0.0
    frame_times = []

    try:
        while True:
            ret, frame, ts_ms = camera.read()
            if not ret or frame is None:
                break

            now = time.time()
            frame_times.append(now)
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) > 1:
                rolling_fps = len(frame_times) / (frame_times[-1] - frame_times[0])

            # Process hand tracking
            hands, latency_ms = app.tracker.process_frame(frame, ts_ms)
            latencies.append(latency_ms)

            # Store lightweight record for quality assessment
            from motion_mentor.storage.models import LandmarkFrameRecord
            records.append(
                LandmarkFrameRecord(
                    session_id="test",
                    frame_index=len(records),
                    timestamp_ms=ts_ms,
                    hands=hands,
                    valid=len(hands) > 0,
                )
            )

            # Render HUD
            hud_frame = frame.copy()
            app.hud.render(
                hud_frame,
                hands=hands,
                fps=rolling_fps,
                latency_ms=latency_ms,
                dropped_frames=getattr(camera, "dropped_frame_count", 0),
                status="preview",
            )

            if not args.headless:
                cv2.imshow(window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.duration and (time.time() - t_start) >= args.duration:
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Camera test stopped by user (Ctrl-C).[/yellow]")
    finally:
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()
        app.close()

    # Generate and print Quality Report
    summary = app.quality_evaluator.evaluate(
        records=records,
        latencies_ms=latencies,
        dropped_frames=getattr(camera, "dropped_frame_count", 0),
    )
    print_terminal_quality_report(summary)


def cmd_record(args: argparse.Namespace) -> None:
    """Record an expert or trainee demonstration."""
    app = MotionMentorApp(args.config)
    activity = app.load_or_create_activity(args.activity)

    console.print(f"[bold green]Starting Recording Session for Activity: {activity.name}[/bold green]")
    console.print(f"Role: [bold]{args.role.upper()}[/bold] | Participant: {args.participant}")

    camera = app.create_camera(
        camera_id=args.camera_id,
        use_synthetic=args.synthetic,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    session_id = generate_uuid()
    video_out = Path(f"data/recordings/{session_id}.mp4")
    landmarks_out = Path(f"data/landmarks/{session_id}.parquet")

    recorder = SessionRecorder(
        session_id=session_id,
        activity=activity,
        role=args.role,
        participant_id=args.participant,
        camera_id=str(args.camera_id),
        width=args.width,
        height=args.height,
        fps=args.fps,
        video_output_path=video_out,
        landmark_output_path=landmarks_out,
    )

    window_name = "MotionMentor Recording (Space: Record/Stop, 'q': Quit)"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    latencies = []
    state = "countdown" if args.countdown > 0 else "recording"
    countdown_start = time.time()
    recording_start = 0.0

    if state == "recording":
        recorder.start()
        recording_start = time.time()

    rolling_fps = 30.0
    frame_times = []

    try:
        while True:
            ret, frame, ts_ms = camera.read()
            if not ret or frame is None:
                break

            now = time.time()
            frame_times.append(now)
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) > 1:
                rolling_fps = len(frame_times) / (frame_times[-1] - frame_times[0])

            hands, latency_ms = app.tracker.process_frame(frame, ts_ms)
            if state == "recording":
                latencies.append(latency_ms)
                recorder.record_frame(frame, ts_ms, hands)

            # State transitions
            hud_frame = frame.copy()
            countdown_left = 0
            rec_elapsed = 0.0

            if state == "countdown":
                elapsed = time.time() - countdown_start
                countdown_left = max(0, int(args.countdown - elapsed) + 1)
                if elapsed >= args.countdown:
                    state = "recording"
                    recorder.start()
                    recording_start = time.time()
            elif state == "recording":
                rec_elapsed = time.time() - recording_start
                if args.duration and rec_elapsed >= args.duration:
                    break

            app.hud.render(
                hud_frame,
                hands=hands,
                fps=rolling_fps,
                latency_ms=latency_ms,
                dropped_frames=getattr(camera, "dropped_frame_count", 0),
                status=state,
                countdown_sec=countdown_left,
                recording_time_sec=rec_elapsed,
                activity_title=activity.name,
                role=args.role,
            )

            if not args.headless:
                cv2.imshow(window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == 32:  # Spacebar toggle
                    if state == "recording":
                        break
                    elif state == "countdown":
                        state = "recording"
                        recorder.start()
                        recording_start = time.time()

    except KeyboardInterrupt:
        console.print("\n[yellow]Recording stopped by user (Ctrl-C). Finalizing session...[/yellow]")
    finally:
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()

    # Finish and persist
    summary = app.quality_evaluator.evaluate(
        records=recorder.frame_records,
        latencies_ms=latencies,
        dropped_frames=getattr(camera, "dropped_frame_count", 0),
    )
    session, records = recorder.finish(quality_summary=summary)
    app.db.save_session(session)
    app.close()

    console.print(f"\n[bold green]Session recorded successfully![/bold green]")
    console.print(f"Session ID: [cyan]{session.session_id}[/cyan]")
    console.print(f"Video saved to: [yellow]{session.video_path}[/yellow]")
    console.print(f"Landmarks saved to: [yellow]{session.landmark_path}[/yellow]")
    print_terminal_quality_report(summary, session.session_id, session.activity_name)


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay a recorded session with synchronized skeleton overlay."""
    app = MotionMentorApp(args.config)
    session = app.db.get_session(args.session)
    if not session:
        console.print(f"[bold red]Session not found in database: {args.session}[/bold red]")
        sys.exit(1)

    if not session.video_path or not Path(session.video_path).exists():
        console.print(f"[bold red]Video file missing: {session.video_path}[/bold red]")
        sys.exit(1)

    if not session.landmark_path or not Path(session.landmark_path).exists():
        console.print(f"[bold red]Landmarks file missing: {session.landmark_path}[/bold red]")
        sys.exit(1)

    records = load_landmarks_parquet(session.landmark_path)
    records_by_idx = {r.frame_index: r for r in records}

    cap = cv2.VideoCapture(session.video_path)
    fps = session.nominal_fps or 30.0
    delay_ms = int(1000.0 / fps)

    window_name = f"Replay: {session.activity_name} ({session.role}) - Space: Pause, 'q': Quit"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Loop video or break
                    if args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    break

                rec = records_by_idx.get(frame_idx)
                hands = rec.hands if rec else []

                t_sec = frame_idx / fps
                app.hud.render(
                    frame,
                    hands=hands,
                    fps=fps,
                    latency_ms=session.quality_summary.median_latency_ms if session.quality_summary else 0.0,
                    status="replaying",
                    recording_time_sec=t_sec,
                )
                frame_idx += 1

            if not args.headless:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == 32:  # Spacebar pause
                    paused = not paused
            else:
                # Headless verification mode
                if args.duration and (frame_idx / fps) >= args.duration:
                    break

    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        app.close()

    console.print(f"[bold green]Replay completed ({frame_idx} frames verified).[/bold green]")


def cmd_list_sessions(args: argparse.Namespace) -> None:
    """List sessions stored in the SQLite database."""
    app = MotionMentorApp(args.config)
    sessions = app.db.list_sessions(activity_id=args.activity, role=args.role)

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="MotionMentor Stored Sessions", header_style="bold cyan")
    table.add_column("Session ID", style="cyan")
    table.add_column("Activity", style="bold")
    table.add_column("Role", style="magenta")
    table.add_column("Duration", justify="right")
    table.add_column("Frames", justify="right")
    table.add_column("Hand Cov", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("Status", justify="center")

    for s in sessions:
        cov_str = f"{s.quality_summary.detection_coverage_pct:.0f}%" if s.quality_summary else "-"
        fps_str = f"{s.quality_summary.effective_fps:.1f}" if s.quality_summary else f"{s.nominal_fps:.0f}"
        status_str = (
            "[green]PASS[/green]"
            if s.quality_summary and s.quality_summary.meets_criteria
            else "[yellow]REVIEW[/yellow]"
        )
        table.add_row(
            s.session_id[:8] + "...",
            s.activity_name or s.activity_id,
            s.role,
            f"{s.duration_seconds:.1f}s",
            str(s.total_frames),
            cov_str,
            fps_str,
            status_str,
        )

    console.print(table)


def cmd_export(args: argparse.Namespace) -> None:
    """Export a session and its landmarks to a standalone JSON file."""
    app = MotionMentorApp(args.config)
    session = app.db.get_session(args.session)
    if not session:
        console.print(f"[bold red]Session {args.session} not found.[/bold red]")
        sys.exit(1)

    records = load_landmarks_parquet(session.landmark_path)
    out_path = Path(args.output or f"data/{session.session_id}_export.json")
    export_session_json(session, records, out_path)
    console.print(f"[bold green]Exported session to {out_path}[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="motion-mentor",
        description="MotionMentor - Explainable Hand Motion Skill Assessment System",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # test-camera
    p_test = subparsers.add_parser("test-camera", help="Test camera feed, skeleton tracking, and HUD")
    p_test.add_argument("--camera-id", default=0, help="Camera device index (default: 0)")
    p_test.add_argument("--synthetic", action="store_true", help="Use synthetic animated test pattern")
    p_test.add_argument("--headless", action="store_true", help="Run without opening GUI preview window")
    p_test.add_argument("--duration", type=float, default=None, help="Auto exit after N seconds")
    p_test.add_argument("--width", type=int, default=1280)
    p_test.add_argument("--height", type=int, default=720)
    p_test.add_argument("--fps", type=int, default=30)

    # record
    p_rec = subparsers.add_parser("record", help="Record an expert or trainee session")
    p_rec.add_argument("--activity", default="reach_and_pinch", help="Activity name or YAML path")
    p_rec.add_argument("--role", choices=["expert", "trainee"], default="expert")
    p_rec.add_argument("--participant", default="local-user", help="Participant identifier")
    p_rec.add_argument("--countdown", type=int, default=3, help="Countdown seconds before recording")
    p_rec.add_argument("--duration", type=float, default=None, help="Stop after N seconds")
    p_rec.add_argument("--camera-id", default=0)
    p_rec.add_argument("--synthetic", action="store_true")
    p_rec.add_argument("--headless", action="store_true")
    p_rec.add_argument("--width", type=int, default=1280)
    p_rec.add_argument("--height", type=int, default=720)
    p_rec.add_argument("--fps", type=int, default=30)

    # replay
    p_rep = subparsers.add_parser("replay", help="Replay a recorded session with landmark overlay")
    p_rep.add_argument("--session", required=True, help="Session UUID")
    p_rep.add_argument("--loop", action="store_true", help="Loop playback continuously")
    p_rep.add_argument("--headless", action="store_true")
    p_rep.add_argument("--duration", type=float, default=None)

    # list-sessions
    p_ls = subparsers.add_parser("list-sessions", help="List stored sessions")
    p_ls.add_argument("--activity", default=None)
    p_ls.add_argument("--role", default=None)

    # export
    p_exp = subparsers.add_parser("export", help="Export session and landmarks to JSON")
    p_exp.add_argument("--session", required=True)
    p_exp.add_argument("--output", default=None)

    args = parser.parse_args()

    commands = {
        "test-camera": cmd_test_camera,
        "record": cmd_record,
        "replay": cmd_replay,
        "list-sessions": cmd_list_sessions,
        "export": cmd_export,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)


if __name__ == "__main__":
    main()
