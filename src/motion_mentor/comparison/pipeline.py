"""End-to-end comparison pipeline: feature prep, DTW alignment, scoring, feedback."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from motion_mentor.app import MotionMentorApp
from motion_mentor.comparison.dtw import align_sequences, constrained_dtw
from motion_mentor.comparison.feedback import FeedbackGenerator
from motion_mentor.comparison.reference import CORE_ALIGNMENT_FEATURES, alignment_matrix
from motion_mentor.comparison.scoring import (
    ScoringEngine,
    determine_interpretation_band,
    scored_frame_fraction,
)
from motion_mentor.processing.features import extract_session_features_df
from motion_mentor.processing.normalization import normalize_session_records
from motion_mentor.processing.smoothing import smooth_landmark_records
from motion_mentor.storage.files import (
    load_features_parquet,
    load_landmarks_parquet,
    save_features_parquet,
)
from motion_mentor.storage.models import (
    Activity,
    AssessmentResult,
    ComponentScores,
    QualitySummary,
    Session,
)

console = Console()


def prepare_session_features(app: MotionMentorApp, session: Session) -> pd.DataFrame:
    """Ensure features DataFrame is computed and available for a session."""
    feat_path = Path(f"data/features/{session.session_id}_features.parquet")
    if feat_path.exists():
        return load_features_parquet(feat_path)

    if not session.landmark_path or not Path(session.landmark_path).exists():
        raise FileNotFoundError(f"Landmarks missing for session: {session.session_id}")

    raw_records = load_landmarks_parquet(session.landmark_path)
    smoothed = smooth_landmark_records(raw_records)
    normalized, global_wrist, global_orient = normalize_session_records(smoothed)
    df_features = extract_session_features_df(
        normalized, global_trajectory=global_wrist, global_orientations=global_orient
    )

    save_features_parquet(df_features, feat_path)
    return df_features


class EnvelopeScore(NamedTuple):
    """One attempt measured against one reference envelope."""

    component_scores: ComponentScores
    overall_score: float
    critical_failures: List[str]
    per_feature_z: Dict[str, float]
    aligned_ref_df: pd.DataFrame
    aligned_trainee_df: pd.DataFrame
    warping_path: List[Tuple[int, int]]
    dtw_distance: float
    scored_fraction: float


def score_attempt_against_envelope(
    activity: Activity,
    ref_df: pd.DataFrame,
    trainee_df: pd.DataFrame,
    trainee_duration_sec: float,
    reference_duration_sec: float,
    trainee_quality: Optional[QualitySummary] = None,
) -> EnvelopeScore:
    """
    Align an attempt to a reference envelope and score it.

    Separated from compare_attempt_to_reference so calibration work can score against a
    throwaway envelope without writing profiles or assessments to the database.
    """
    align_cols = CORE_ALIGNMENT_FEATURES
    ref_align_cols = [f"{c}_mean" for c in align_cols if f"{c}_mean" in ref_df.columns]
    trainee_align_cols = [c for c in align_cols if c in trainee_df.columns]

    # Untracked frames are NaN. Bridge them for alignment only; the scoring below reads
    # the original columns, so a gap stays a gap where it matters.
    seq_ref = alignment_matrix(ref_df, ref_align_cols)
    seq_trainee = alignment_matrix(trainee_df, trainee_align_cols)

    dtw_dist, warping_path, _ = constrained_dtw(seq_ref, seq_trainee, window_ratio=0.25)

    aligned_ref, aligned_trainee = align_sequences(ref_df.to_numpy(), trainee_df.to_numpy(), warping_path)
    aligned_ref_df = pd.DataFrame(aligned_ref, columns=ref_df.columns)
    aligned_trainee_df = pd.DataFrame(aligned_trainee, columns=trainee_df.columns)

    comp_scores, overall_score, critical_fails, per_feature_z = ScoringEngine(activity).evaluate_attempt(
        aligned_ref_df=aligned_ref_df,
        aligned_trainee_df=aligned_trainee_df,
        trainee_duration_sec=trainee_duration_sec,
        reference_duration_sec=reference_duration_sec,
        warping_path=warping_path,
        trainee_quality=trainee_quality,
    )

    return EnvelopeScore(
        component_scores=comp_scores,
        overall_score=overall_score,
        critical_failures=critical_fails,
        per_feature_z=per_feature_z,
        aligned_ref_df=aligned_ref_df,
        aligned_trainee_df=aligned_trainee_df,
        warping_path=warping_path,
        dtw_distance=dtw_dist,
        scored_fraction=scored_frame_fraction(aligned_ref_df, aligned_trainee_df),
    )


def compare_attempt_to_reference(
    app: MotionMentorApp,
    attempt_session: Session,
    reference_session_or_profile_id: str,
) -> AssessmentResult:
    """Run end-to-end DTW alignment, scoring, and feedback generation."""
    activity = app.db.get_activity(attempt_session.activity_id) or app.load_or_create_activity(attempt_session.activity_name)

    # 1. Load or Build Reference Profile
    ref_profile = app.db.get_reference_profile(reference_session_or_profile_id)
    ref_df: pd.DataFrame

    if not (ref_profile and Path(ref_profile.profile_path).exists()):
        # Fall back to the latest profile for this activity. A reference is a deliberate
        # artifact: nothing here builds one on the fly. A single-take profile built as a
        # side effect of a comparison has no measured tolerance at all, only the floors,
        # and it looks identical to a real one in the UI.
        ref_profile = app.db.get_latest_reference_profile(activity.activity_id)
        if not (ref_profile and Path(ref_profile.profile_path).exists()):
            raise ValueError(
                f"No reference profile found for activity '{activity.activity_id}' "
                f"(looked up '{reference_session_or_profile_id}'). Build one first with: "
                f"motion-mentor build-reference --activity {activity.activity_id}"
            )

    ref_df = pd.read_parquet(ref_profile.profile_path)
    ref_id = ref_profile.reference_id
    ref_duration = ref_profile.duration_mean_sec

    # 2. Prepare Trainee Features
    trainee_df = prepare_session_features(app, attempt_session)

    # 3 & 4. Align and score
    result = score_attempt_against_envelope(
        activity=activity,
        ref_df=ref_df,
        trainee_df=trainee_df,
        trainee_duration_sec=attempt_session.duration_seconds,
        reference_duration_sec=ref_duration,
        trainee_quality=attempt_session.quality_summary,
    )
    aligned_ref_df = result.aligned_ref_df
    aligned_trainee_df = result.aligned_trainee_df
    comp_scores = result.component_scores
    overall_score = result.overall_score
    critical_fails = result.critical_failures
    per_feature_z = result.per_feature_z

    # 5. Coaching Feedback
    feedback_gen = FeedbackGenerator()
    feedback_items = feedback_gen.generate_feedback(
        component_scores=comp_scores,
        per_feature_z=per_feature_z,
        aligned_ref_df=aligned_ref_df,
        aligned_trainee_df=aligned_trainee_df,
        critical_failures=critical_fails,
    )

    band = determine_interpretation_band(overall_score, critical_fails)
    reliability = "high"
    if attempt_session.quality_summary and attempt_session.quality_summary.detection_coverage_pct < 85.0:
        reliability = "low"
    # A score built from a handful of usable frames is not a confident score, however
    # good the number looks.
    if result.scored_fraction < 0.85:
        reliability = "low"
        console.print(
            f"[yellow]Only {result.scored_fraction * 100:.0f}% of aligned frames were scorable "
            f"(missing tracking or thin reference support).[/yellow]"
        )

    assessment = AssessmentResult(
        attempt_session_id=attempt_session.session_id,
        reference_profile_id=ref_id,
        activity_name=activity.name,
        overall_score=overall_score,
        reliability=reliability,
        interpretation_band=band,
        component_scores=comp_scores,
        critical_failures=critical_fails,
        feedback=feedback_items,
    )

    app.db.save_assessment(assessment)
    return assessment


def print_assessment_scorecard(assessment: AssessmentResult) -> None:
    """Print complete assessment result scorecard to terminal."""
    console.print()
    band_colors = {
        "Excellent match": "bold green",
        "Good match": "bold cyan",
        "Developing": "bold yellow",
        "Needs review": "bold red",
    }
    b_color = band_colors.get(assessment.interpretation_band, "bold white")

    # Header Panel
    console.print(
        Panel(
            f"ACTIVITY: [bold]{assessment.activity_name.upper()}[/bold]\n"
            f"OVERALL SCORE: [{b_color}]{assessment.overall_score:.1f} / 100[/{b_color}]  "
            f"({assessment.interpretation_band})  |  Reliability: [bold]{assessment.reliability.upper()}[/bold]",
            title="MotionMentor Skill Assessment Report",
            style="cyan",
            expand=False,
        )
    )

    # Component Scores Table
    table = Table(title="Component Scores Breakdown", header_style="bold cyan")
    table.add_column("Component", style="bold")
    table.add_column("Score (0-100)", justify="right")
    table.add_column("Weight", justify="right", style="dim")
    table.add_column("Evaluation", justify="center")

    def score_status(val: float) -> str:
        if val >= 90.0:
            return "[green]Excellent[/green]"
        elif val >= 80.0:
            return "[cyan]Good[/cyan]"
        elif val >= 70.0:
            return "[yellow]Developing[/yellow]"
        else:
            return "[red]Needs Review[/red]"

    c = assessment.component_scores
    table.add_row("Local Hand Pose", f"{c.pose:.1f}", "25%", score_status(c.pose))
    table.add_row("Global Trajectory", f"{c.trajectory:.1f}", "25%", score_status(c.trajectory))
    table.add_row("Hand Orientation", f"{c.orientation:.1f}", "15%", score_status(c.orientation))
    table.add_row("Timing & Duration", f"{c.timing:.1f}", "15%", score_status(c.timing))
    table.add_row("Speed & Smoothness", f"{c.smoothness:.1f}", "10%", score_status(c.smoothness))
    table.add_row("Sequence & Checkpoints", f"{c.sequence:.1f}", "10%", score_status(c.sequence))
    console.print(table)

    # Critical Failures
    if assessment.critical_failures:
        console.print("\n[bold red]CRITICAL CHECKPOINT FAILURES:[/bold red]")
        for fail in assessment.critical_failures:
            console.print(f" [red]✗[/red] {fail}")

    # Actionable Coaching Feedback
    console.print("\n[bold cyan]Actionable Coaching & Corrections:[/bold cyan]")
    for idx, item in enumerate(assessment.feedback, 1):
        sev_color = "red" if item.severity == "high" else "yellow" if item.severity == "medium" else "green"
        console.print(
            f" [bold {sev_color}]{idx}. [{item.component.upper()}][/bold {sev_color}] "
            f"{item.message}"
        )
        if item.recommendation:
            console.print(f"    [dim]Tip:[/dim] {item.recommendation}")
    console.print()
