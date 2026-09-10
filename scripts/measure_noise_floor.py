"""Measure MediaPipe's own jitter from a static-hold recording.

The tolerance floors in ``motion_mentor.comparison.reference.MIN_FEATURE_TOLERANCES``
say how much variation is too small to be meaningful. They are currently guessed
constants. This script replaces the guess with a measurement.

Record the expert holding one pose, motionless, for about five seconds, at the same
camera distance, lighting and resolution used for real captures. The hand is not
moving, so any frame-to-frame variation in the derived features is the tracker's
noise and nothing else. That is the floor.

Usage:
    uv run python scripts/measure_noise_floor.py <session_id>

The numbers are camera- and setup-specific. Re-measure whenever the rig changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motion_mentor.comparison.reference import (  # noqa: E402
    MIN_FEATURE_TOLERANCES,
    get_feature_min_tolerance,
)

CATEGORY_OF_KEY = {
    "angle_default": "joint angles, pitch/roll/yaw (degrees)",
    "distance_default": "distances and spreads (normalized)",
    "position_default": "wrist positions (normalized)",
    "normal_default": "palm normal components",
    "velocity_default": "velocity, acceleration, jerk",
}


def category_key(feature: str) -> str:
    """Reverse the dispatch in get_feature_min_tolerance to name a feature's category."""
    tolerance = get_feature_min_tolerance(feature)
    for key, value in MIN_FEATURE_TOLERANCES.items():
        if value == tolerance:
            return key
    return "angle_default"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_id", help="Session id of the static-hold recording")
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile of per-feature std to use as the category floor (default 90)",
    )
    args = parser.parse_args()

    feat_path = Path(f"data/features/{args.session_id}_features.parquet")
    if not feat_path.exists():
        sys.exit(f"No features for {args.session_id}. Run: motion-mentor process --session {args.session_id}")

    df = pd.read_parquet(feat_path)
    if "valid" in df.columns:
        df = df[df["valid"].astype(bool)]
    if len(df) < 15:
        sys.exit(f"Only {len(df)} tracked frames. Record a longer static hold with better tracking.")

    print(f"Static-hold session {args.session_id}: {len(df)} tracked frames\n")

    by_category: dict[str, list[tuple[str, float]]] = {key: [] for key in MIN_FEATURE_TOLERANCES}
    feature_cols = [c for c in df.columns if c not in ("frame_index", "timestamp_ms", "valid")]
    for col in feature_cols:
        values = df[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 15:
            continue
        by_category[category_key(col)].append((col, float(np.std(values))))

    print(f"{'feature':<24}{'measured std':>14}{'current floor':>16}")
    print("-" * 54)
    suggestions: dict[str, float] = {}
    for key, entries in by_category.items():
        if not entries:
            continue
        print(f"\n{CATEGORY_OF_KEY[key]}")
        for name, std in sorted(entries, key=lambda e: -e[1]):
            print(f"  {name:<22}{std:>14.4f}{MIN_FEATURE_TOLERANCES[key]:>16.4f}")
        stds = [std for _, std in entries]
        # A floor at the high end of the category, so the noisiest feature in it is still
        # covered. The mean would leave half the category scoring its own jitter.
        suggestions[key] = float(np.percentile(stds, args.percentile))

    print("\n\nSuggested MIN_FEATURE_TOLERANCES "
          f"(p{args.percentile:.0f} of each category, from session {args.session_id}):\n")
    for key in MIN_FEATURE_TOLERANCES:
        if key in suggestions:
            current = MIN_FEATURE_TOLERANCES[key]
            new = suggestions[key]
            direction = "looser" if new > current else "tighter"
            print(f'    "{key}": {new:.4f},  # was {current} ({direction})')
        else:
            print(f'    "{key}": {MIN_FEATURE_TOLERANCES[key]},  # unchanged, no features measured')

    print("\nPaste these into src/motion_mentor/comparison/reference.py and note the")
    print("session id and date in the comment above them, then rebuild every reference.")


if __name__ == "__main__":
    main()
