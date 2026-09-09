#!/usr/bin/env python3
"""Standalone script: Session Replay with Landmark Overlay."""

import sys
from pathlib import Path

# Add src to sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
from motion_mentor.cli import cmd_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionMentor - Replay Session")
    parser.add_argument("--session", required=True, help="Session UUID to replay")
    parser.add_argument("--loop", action="store_true", help="Loop playback continuously")
    parser.add_argument("--headless", action="store_true", help="Run headlessly")
    parser.add_argument("--duration", type=float, default=None, help="Auto exit after N seconds")
    parser.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()
    cmd_replay(args)


if __name__ == "__main__":
    main()
