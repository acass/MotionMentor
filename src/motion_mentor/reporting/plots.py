"""Multi-panel kinematic and geometric plots of extracted session features."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

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
