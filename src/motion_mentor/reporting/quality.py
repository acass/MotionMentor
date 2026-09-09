"""Quality assessment engine and reporting for camera and tracking capture."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from motion_mentor.storage.models import LandmarkFrameRecord, QualitySummary, Session


class QualityEvaluator:
    """Evaluates capture fidelity against PRD Phase 0 and Phase 1 exit criteria."""

    def __init__(
        self,
        min_hand_coverage_pct: float = 90.0,
        min_median_fps: float = 24.0,
        max_p95_latency_ms: float = 45.0,
    ) -> None:
        self.min_hand_coverage = min_hand_coverage_pct
        self.min_median_fps = min_median_fps
        self.max_p95_latency = max_p95_latency_ms

    def evaluate(
        self,
        records: List[LandmarkFrameRecord],
        latencies_ms: Optional[List[float]] = None,
        dropped_frames: int = 0,
    ) -> QualitySummary:
        total_frames = len(records)
        if total_frames == 0:
            return QualitySummary(
                total_frames=0,
                meets_criteria=False,
                status_reasons=["No frames captured in session."],
            )

        frames_with_hand = sum(1 for r in records if len(r.hands) > 0)
        coverage_pct = round((frames_with_hand / total_frames) * 100.0, 1)

        # Timestamps and FPS calculation
        timestamps = [r.timestamp_ms for r in records]
        effective_fps = 0.0
        jitter_std_ms = 0.0

        if len(timestamps) > 1:
            diffs_ms = np.diff(timestamps)
            # Filter out non-positive deltas if any
            valid_diffs = diffs_ms[diffs_ms > 0]
            if len(valid_diffs) > 0:
                median_interval_sec = np.median(valid_diffs) / 1000.0
                if median_interval_sec > 0:
                    effective_fps = round(float(1.0 / median_interval_sec), 1)
                jitter_std_ms = round(float(np.std(valid_diffs)), 2)

        # Latencies
        median_lat = 0.0
        p95_lat = 0.0
        if latencies_ms and len(latencies_ms) > 0:
            median_lat = round(float(np.median(latencies_ms)), 1)
            p95_lat = round(float(np.percentile(latencies_ms, 95)), 1)

        issues: List[str] = []
        meets_criteria = True

        if coverage_pct < self.min_hand_coverage:
            meets_criteria = False
            issues.append(
                f"Hand detection coverage ({coverage_pct}%) is below the required {self.min_hand_coverage}%."
            )

        if effective_fps < self.min_median_fps:
            meets_criteria = False
            issues.append(
                f"Effective FPS ({effective_fps}) is below the required minimum of {self.min_median_fps} FPS."
            )

        if p95_lat > self.max_p95_latency:
            issues.append(
                f"P95 inference latency ({p95_lat}ms) exceeds the target {self.max_p95_latency}ms threshold."
            )

        if dropped_frames > (total_frames * 0.05):
            meets_criteria = False
            issues.append(f"Dropped frames count ({dropped_frames}) is excessive.")

        return QualitySummary(
            total_frames=total_frames,
            frames_with_hand=frames_with_hand,
            detection_coverage_pct=coverage_pct,
            dropped_frames=dropped_frames,
            effective_fps=effective_fps,
            median_latency_ms=median_lat,
            p95_latency_ms=p95_lat,
            jitter_std_ms=jitter_std_ms,
            meets_criteria=meets_criteria,
            status_reasons=issues,
        )


def print_terminal_quality_report(
    summary: QualitySummary,
    session_id: Optional[str] = None,
    activity_name: Optional[str] = None,
) -> None:
    """Print an attractive summary table to stdout using rich."""
    console = Console()

    table = Table(title="MotionMentor Capture Quality Report", header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Observed Value", style="white")
    table.add_column("PRD Requirement", style="dim")
    table.add_column("Status", justify="center")

    def status_tag(passed: bool) -> str:
        return "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"

    table.add_row(
        "Total Frames Captured",
        str(summary.total_frames),
        "-",
        "[bold green]OK[/bold green]",
    )

    cov_pass = summary.detection_coverage_pct >= 90.0
    table.add_row(
        "Hand Detection Coverage",
        f"{summary.detection_coverage_pct:.1f}% ({summary.frames_with_hand}/{summary.total_frames})",
        ">= 90.0%",
        status_tag(cov_pass),
    )

    fps_pass = summary.effective_fps >= 24.0
    table.add_row(
        "Effective Frame Rate",
        f"{summary.effective_fps:.1f} FPS",
        ">= 24.0 FPS",
        status_tag(fps_pass),
    )

    lat_pass = summary.median_latency_ms <= 40.0
    table.add_row(
        "Median Inference Latency",
        f"{summary.median_latency_ms:.1f} ms",
        "<= 40.0 ms",
        status_tag(lat_pass),
    )

    table.add_row(
        "P95 Inference Latency",
        f"{summary.p95_latency_ms:.1f} ms",
        "<= 50.0 ms",
        "[dim]INFO[/dim]",
    )

    drop_pass = summary.dropped_frames <= max(2, int(summary.total_frames * 0.05))
    table.add_row(
        "Dropped Frames",
        str(summary.dropped_frames),
        "< 5% of total",
        status_tag(drop_pass),
    )

    table.add_row(
        "Frame Interval Jitter (std)",
        f"{summary.jitter_std_ms:.2f} ms",
        "-",
        "[dim]INFO[/dim]",
    )

    overall_text = (
        "[bold green]CRITERIA MET — Ready for Session Replay & Comparison[/bold green]"
        if summary.meets_criteria
        else "[bold red]CRITERIA NOT MET — Review Camera Setup & Lighting[/bold red]"
    )

    console.print()
    console.print(table)
    console.print(Panel(overall_text, expand=False))
    if summary.status_reasons:
        console.print("[yellow]Warnings & Issues:[/yellow]")
        for reason in summary.status_reasons:
            console.print(f" • {reason}")
    console.print()


def generate_markdown_quality_report(
    session: Session,
    summary: QualitySummary,
) -> str:
    """Generate markdown formatted report suitable for documentation or artifact logs."""
    status_str = "PASS" if summary.meets_criteria else "NEEDS REVIEW"
    md = f"""# Capture Quality Report: {session.session_id}

- **Activity:** {session.activity_name or session.activity_id} (v{session.activity_version})
- **Role:** {session.role.capitalize()}
- **Duration:** {session.duration_seconds}s ({summary.total_frames} frames)
- **Resolution:** {session.resolution[0]}x{session.resolution[1]} @ {session.nominal_fps} FPS nominal
- **Result:** **{status_str}**

## Metrics Summary

| Metric | Measured | Target | Verdict |
|---|---|---|---|
| Hand Coverage | {summary.detection_coverage_pct:.1f}% | >= 90% | {'✅ PASS' if summary.detection_coverage_pct >= 90.0 else '❌ FAIL'} |
| Effective FPS | {summary.effective_fps:.1f} FPS | >= 24 FPS | {'✅ PASS' if summary.effective_fps >= 24.0 else '❌ FAIL'} |
| Median Latency | {summary.median_latency_ms:.1f} ms | <= 40 ms | {'✅ PASS' if summary.median_latency_ms <= 40.0 else '⚠️ WARN'} |
| P95 Latency | {summary.p95_latency_ms:.1f} ms | <= 50 ms | {'✅ PASS' if summary.p95_latency_ms <= 50.0 else '⚠️ WARN'} |
| Dropped Frames | {summary.dropped_frames} | < 5% | {'✅ PASS' if summary.dropped_frames <= 5 else '❌ FAIL'} |
| Jitter (Std) | {summary.jitter_std_ms:.2f} ms | - | Info |

"""
    if summary.status_reasons:
        md += "## Issues & Recommendations\n\n"
        for r in summary.status_reasons:
            md += f"- {r}\n"
    return md
