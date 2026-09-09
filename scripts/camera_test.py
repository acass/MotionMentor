#!/usr/bin/env python3
"""Standalone script: Camera Test & Feasibility Validator."""

import sys
from pathlib import Path

# Add src to sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
from motion_mentor.cli import cmd_test_camera


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionMentor - Camera Test & Feasibility Validator")
    parser.add_argument("--camera-id", default=0, help="Camera index (default: 0)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic animated pattern")
    parser.add_argument("--headless", action="store_true", help="Headless test without GUI window")
    parser.add_argument("--duration", type=float, default=None, help="Auto exit after N seconds")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()
    cmd_test_camera(args)


if __name__ == "__main__":
    main()
