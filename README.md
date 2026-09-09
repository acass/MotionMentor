# MotionMentor

Explainable hand motion skill assessment and procedural coaching system using computer vision.

## Overview

MotionMentor captures hand movements from a standard camera, extracts 21 landmarks per hand, computes explainable geometric and kinematic features, and compares attempts against expert reference profiles using constrained Dynamic Time Warping (DTW).

## Quick Start

```bash
# Test camera and preview skeleton overlay
uv run python scripts/camera_test.py

# Record a new session (expert or trainee)
uv run python scripts/record_session.py --activity reach_and_pinch --role expert

# Replay a recorded session
uv run python scripts/replay_session.py --session <SESSION_ID>

# Run test suite
uv run pytest
```
