#!/usr/bin/env python3
"""Standalone script: Interactive Session Recorder."""

import sys
from pathlib import Path

# Add src to sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
from motion_mentor.cli import cmd_record


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionMentor - Record Session")
    parser.add_argument("--activity", default="reach_and_pinch", help="Activity name or config file")
    parser.add_argument("--role", choices=["expert", "trainee"], default="expert")
    parser.add_argument("--participant", default="local-user", help="Participant ID")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before recording")
    parser.add_argument("--duration", type=float, default=None, help="Auto stop recording after N seconds")
    parser.add_argument("--camera-id", default=0, help="Camera index")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic camera")
    parser.add_argument("--headless", action="store_true", help="Run headlessly")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()
    cmd_record(args)


if __name__ == "__main__":
    main()
