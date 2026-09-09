#!/usr/bin/env python3
"""Plot extracted motion features (joint angles, pinch aperture, kinematics) over time."""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

from motion_mentor.app import MotionMentorApp
from motion_mentor.processing.features import extract_session_features_df
from motion_mentor.processing.normalization import normalize_session_records
from motion_mentor.processing.smoothing import smooth_landmark_records
from motion_mentor.storage.files import load_features_parquet, load_landmarks_parquet

console = Console()


def plot_session_features(
    df: pd.DataFrame,
    session_id: str,
    output_png: str | Path | None = None,
    show_plot: bool = True,
) -> None:
    """Generate multi-panel kinematic and geometric plot."""
    if df.empty:
        console.print("[red]DataFrame is empty. Nothing to plot.[/red]")
        return

    time_sec = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f"MotionMentor Feature Extraction: {session_id}", fontsize=14, fontweight="bold")

    # Panel 1: Wrist Path
    ax1 = axes[0]
    ax1.plot(time_sec, df["wrist_x"], label="Wrist X", color="#00e5ff", linewidth=1.5)
    ax1.plot(time_sec, df["wrist_y"], label="Wrist Y", color="#ff7043", linewidth=1.5)
    ax1.set_ylabel("Position (norm)")
    ax1.set_title("Global Wrist Trajectory", fontsize=11, loc="left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Panel 2: Key Finger Joint Angles
    ax2 = axes[1]
    for angle_col, color, label in [
        ("index_mcp", "#29b6f6", "Index MCP"),
        ("index_pip", "#ab47bc", "Index PIP"),
        ("thumb_mcp", "#26a69a", "Thumb MCP"),
        ("thumb_ip", "#ffa726", "Thumb IP"),
    ]:
        if angle_col in df.columns:
            ax2.plot(time_sec, df[angle_col], label=label, color=color, linewidth=1.5)
    ax2.set_ylabel("Angle (deg)")
    ax2.set_title("Joint Angles (Flexion / Extension)", fontsize=11, loc="left")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    # Panel 3: Pinch Aperture & Pinch Velocity
    ax3 = axes[2]
    if "pinch_distance" in df.columns:
        ax3.plot(time_sec, df["pinch_distance"], label="Pinch Distance (norm)", color="#66bb6a", linewidth=2.0)
    ax3.set_ylabel("Distance")
    ax3.set_title("Thumb-to-Index Pinch Aperture", fontsize=11, loc="left")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")

    # Panel 4: Kinematics (Wrist Velocity and Jerk)
    ax4 = axes[3]
    if "wrist_velocity" in df.columns:
        ax4.plot(time_sec, df["wrist_velocity"], label="Velocity", color="#42a5f5", linewidth=1.5)
    if "wrist_jerk" in df.columns:
        # Scale jerk for visualization
        scaled_jerk = df["wrist_jerk"] / max(1.0, df["wrist_jerk"].max())
        ax4.plot(time_sec, scaled_jerk, label="Jerk (normalized)", color="#ef5350", linewidth=1.0, alpha=0.7)
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Rate")
    ax4.set_title("Kinematics & Smoothness Profile", fontsize=11, loc="left")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right")

    plt.tight_layout()

    if output_png:
        out_path = Path(output_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        console.print(f"[green]Plot saved to {out_path}[/green]")

    if show_plot:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot motion features from a recorded session.")
    parser.add_argument("--session", required=True, help="Session UUID")
    parser.add_argument("--output", default=None, help="Path to save plot PNG")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive plot window")
    parser.add_argument("--smooth", action="store_true", default=True, help="Apply One Euro filter before plotting")
    args = parser.parse_args()

    app = MotionMentorApp()
    session = app.db.get_session(args.session)
    if not session or not session.landmark_path:
        console.print(f"[red]Session {args.session} not found in database.[/red]")
        sys.exit(1)

    raw_records = load_landmarks_parquet(session.landmark_path)
    records = raw_records
    if args.smooth:
        records = smooth_landmark_records(records)
    records, global_wrist = normalize_session_records(records)

    df_features = extract_session_features_df(records, global_trajectory=global_wrist)
    plot_session_features(
        df_features,
        session_id=session.session_id,
        output_png=args.output or f"data/features/{session.session_id}_features.png",
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
