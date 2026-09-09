# MotionMentor

Explainable hand motion skill assessment and procedural coaching system using computer vision.

## Overview

MotionMentor captures hand movements from a standard camera, extracts 21 landmarks per hand, computes explainable geometric and kinematic features, and compares attempts against expert reference profiles using constrained Dynamic Time Warping (DTW).

## Quick Start

```bash
# Test camera and preview skeleton overlay
uv run motion-mentor test-camera

# Record a new session (expert or trainee)
uv run motion-mentor record --activity reach_and_pinch --role expert

# Replay a recorded session
uv run motion-mentor replay --session <SESSION_ID>

# Process, build a reference, compare, and serve the dashboard
uv run motion-mentor process --session <SESSION_ID>
uv run motion-mentor build-reference --activity reach_and_pinch
uv run motion-mentor compare --attempt <SESSION_ID> --reference <REF_ID>
uv run motion-mentor serve

# Run test suite
uv run --extra dev pytest
```
