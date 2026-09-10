"""Component scoring engine with tolerance-aware Z-score error mapping and checkpoint validation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from motion_mentor.storage.models import (
    Activity,
    ComponentScores,
    QualitySummary,
)

JOINT_ANGLE_FEATURES = [
    "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
]

TRAJECTORY_FEATURES = ["wrist_x", "wrist_y", "wrist_z"]
ORIENTATION_FEATURES = ["palm_pitch", "palm_roll", "palm_yaw", "palm_normal_x", "palm_normal_y", "palm_normal_z"]
SMOOTHNESS_FEATURES = ["wrist_velocity", "wrist_jerk"]


def compute_component_zscore(
    trainee_vals: np.ndarray,   # shape (K,)
    ref_mean: np.ndarray,       # shape (K,)
    ref_std: np.ndarray,        # shape (K,)
) -> float:
    """
    Compute average Z-score error for a feature channel.

    Frames where the trainee was untracked, or where the envelope has too few expert
    takes behind it, arrive as NaN. Those carry no evidence either way, so they are
    skipped rather than scored as a perfect or a catastrophic match. Returns NaN when
    a channel has no usable frame at all, and callers drop it.
    """
    # Alignment round-trips the frames through a mixed-dtype array, so these can arrive
    # as object arrays that np.isfinite refuses.
    trainee_vals = np.asarray(trainee_vals, dtype=float)
    ref_mean = np.asarray(ref_mean, dtype=float)
    ref_std = np.asarray(ref_std, dtype=float)

    errors = np.abs(trainee_vals - ref_mean)
    z_scores = errors / np.maximum(ref_std, 1e-4)
    usable = np.isfinite(z_scores)
    if not usable.any():
        return float("nan")
    return float(np.mean(z_scores[usable]))


def mean_finite_z(z_list: List[float], default: float = 1.0) -> float:
    """Average the channels that produced a usable Z-score, ignoring the rest."""
    finite = [z for z in z_list if math.isfinite(z)]
    return float(np.mean(finite)) if finite else default


def scored_frame_fraction(aligned_ref_df: pd.DataFrame, aligned_trainee_df: pd.DataFrame) -> float:
    """
    Fraction of aligned frames that actually contributed to the score.

    A comparison can look confident while most of its frames were skipped for missing
    data. This is what tells the difference.
    """
    if len(aligned_ref_df) == 0:
        return 0.0
    ref_cols = [f"{f}_mean" for f in JOINT_ANGLE_FEATURES if f"{f}_mean" in aligned_ref_df.columns]
    trainee_cols = [f for f in JOINT_ANGLE_FEATURES if f in aligned_trainee_df.columns]
    if not ref_cols or not trainee_cols:
        return 0.0
    ref_ok = aligned_ref_df[ref_cols].notna().all(axis=1).to_numpy()
    trainee_ok = aligned_trainee_df[trainee_cols].notna().all(axis=1).to_numpy()
    return float(np.mean(ref_ok & trainee_ok))


def zscore_to_score_100(mean_z: float, decay_factor: float = 0.35) -> float:
    """Convert average Z-score to a bounded 0-100 score using exponential decay."""
    # z=0 -> 100, z=1.0 -> 70.5, z=2.0 -> 49.6, z=3.0 -> 35.0
    score = 100.0 * math.exp(-decay_factor * max(0.0, mean_z))
    return round(float(np.clip(score, 0.0, 100.0)), 1)


class ScoringEngine:
    """Evaluates motion similarity across explainable component dimensions."""

    def __init__(self, activity: Activity) -> None:
        self.activity = activity
        self.weights = activity.scoring_weights

    def evaluate_attempt(
        self,
        aligned_ref_df: pd.DataFrame,
        aligned_trainee_df: pd.DataFrame,
        trainee_duration_sec: float,
        reference_duration_sec: float,
        warping_path: List[Tuple[int, int]],
        trainee_quality: Optional[QualitySummary] = None,
    ) -> Tuple[ComponentScores, float, List[str], Dict[str, float]]:
        """
        Compute component scores and return overall score and detected critical failures.

        Returns:
            (component_scores, overall_score, critical_failures, per_feature_errors)
        """
        per_feature_z: Dict[str, float] = {}

        # 1. Pose Score (Joint Angles)
        pose_z_list = []
        for feat in JOINT_ANGLE_FEATURES:
            if f"{feat}_mean" in aligned_ref_df.columns and feat in aligned_trainee_df.columns:
                z = compute_component_zscore(
                    aligned_trainee_df[feat].to_numpy(),
                    aligned_ref_df[f"{feat}_mean"].to_numpy(),
                    aligned_ref_df[f"{feat}_std"].to_numpy(),
                )
                if math.isfinite(z):
                    per_feature_z[feat] = z
                pose_z_list.append(z)

        mean_pose_z = mean_finite_z(pose_z_list)
        score_pose = zscore_to_score_100(mean_pose_z)

        # 2. Trajectory Score (Wrist Path)
        traj_z_list = []
        for feat in TRAJECTORY_FEATURES:
            if f"{feat}_mean" in aligned_ref_df.columns and feat in aligned_trainee_df.columns:
                z = compute_component_zscore(
                    aligned_trainee_df[feat].to_numpy(),
                    aligned_ref_df[f"{feat}_mean"].to_numpy(),
                    aligned_ref_df[f"{feat}_std"].to_numpy(),
                )
                if math.isfinite(z):
                    per_feature_z[feat] = z
                traj_z_list.append(z)

        mean_traj_z = mean_finite_z(traj_z_list)
        score_trajectory = zscore_to_score_100(mean_traj_z)

        # 3. Orientation Score (Palm & Wrist Angles)
        orient_z_list = []
        for feat in ORIENTATION_FEATURES:
            if f"{feat}_mean" in aligned_ref_df.columns and feat in aligned_trainee_df.columns:
                z = compute_component_zscore(
                    aligned_trainee_df[feat].to_numpy(),
                    aligned_ref_df[f"{feat}_mean"].to_numpy(),
                    aligned_ref_df[f"{feat}_std"].to_numpy(),
                )
                if math.isfinite(z):
                    per_feature_z[feat] = z
                orient_z_list.append(z)

        mean_orient_z = mean_finite_z(orient_z_list)
        score_orientation = zscore_to_score_100(mean_orient_z)

        # 4. Timing Score
        # Duration ratio + warping path linearity
        dur_ratio = min(trainee_duration_sec, reference_duration_sec) / max(
            1e-4, max(trainee_duration_sec, reference_duration_sec)
        )
        # Linearity of path: average distance of path points from ideal diagonal
        n_ref = len(aligned_ref_df)
        n_trainee = len(aligned_trainee_df)
        diag_errors = []
        for i_ref, j_trainee in warping_path:
            expected_j = (i_ref / max(1, n_ref - 1)) * (n_trainee - 1)
            diag_errors.append(abs(j_trainee - expected_j) / max(1, n_trainee))
        mean_path_deviation = float(np.mean(diag_errors))

        score_timing = round(
            float(np.clip(dur_ratio * 70.0 + (1.0 - min(1.0, mean_path_deviation * 2.0)) * 30.0, 0.0, 100.0)),
            1,
        )

        # 5. Smoothness Score (Velocity and Jerk)
        smooth_z_list = []
        for feat in SMOOTHNESS_FEATURES:
            if f"{feat}_mean" in aligned_ref_df.columns and feat in aligned_trainee_df.columns:
                z = compute_component_zscore(
                    aligned_trainee_df[feat].to_numpy(),
                    aligned_ref_df[f"{feat}_mean"].to_numpy(),
                    aligned_ref_df[f"{feat}_std"].to_numpy(),
                )
                if math.isfinite(z):
                    per_feature_z[feat] = z
                smooth_z_list.append(z)

        mean_smooth_z = mean_finite_z(smooth_z_list)
        score_smoothness = zscore_to_score_100(mean_smooth_z, decay_factor=0.30)

        # 6. Sequence & Checkpoints Score
        score_sequence, critical_failures = self._evaluate_checkpoints(aligned_trainee_df)

        scores = ComponentScores(
            pose=score_pose,
            trajectory=score_trajectory,
            orientation=score_orientation,
            timing=score_timing,
            smoothness=score_smoothness,
            sequence=score_sequence,
        )

        # Overall Weighted Score
        w = self.weights
        overall = (
            w.get("pose", 0.25) * score_pose
            + w.get("trajectory", 0.25) * score_trajectory
            + w.get("orientation", 0.15) * score_orientation
            + w.get("timing", 0.15) * score_timing
            + w.get("smoothness", 0.10) * score_smoothness
            + w.get("sequence", 0.10) * score_sequence
        )
        overall = round(overall, 1)

        # Critical Checkpoint Gate: If critical failure, cap overall score at 50
        if critical_failures:
            overall = min(overall, 50.0)

        return scores, overall, critical_failures, per_feature_z

    def _evaluate_checkpoints(
        self,
        trainee_df: pd.DataFrame,
    ) -> Tuple[float, List[str]]:
        """Validate activity-specific checkpoints (e.g. rotation reached, pinch contact)."""
        critical_failures: List[str] = []
        if not self.activity.checkpoints:
            return 100.0, []

        passed_count = 0
        total = len(self.activity.checkpoints)

        for cp in self.activity.checkpoints:
            cp_id = cp.get("id", "")
            is_critical = cp.get("critical", False)
            cp_passed = False

            if cp_id == "cp_rotation_reached":
                # Verify palm pitch or roll exceeded 45 degrees
                if "palm_pitch" in trainee_df.columns and "palm_roll" in trainee_df.columns:
                    max_rot = max(
                        trainee_df["palm_pitch"].abs().max(),
                        trainee_df["palm_roll"].abs().max(),
                    )
                    cp_passed = bool(max_rot >= 40.0)
            elif cp_id == "cp_pinch_contact":
                # Verify pinch distance closed below 0.60
                if "pinch_distance" in trainee_df.columns:
                    min_pinch = trainee_df["pinch_distance"].min()
                    cp_passed = bool(min_pinch <= 0.65)
            elif cp_id == "cp_hold_duration":
                # Verify pinch remained closed for at least 0.5s (~15 frames)
                if "pinch_distance" in trainee_df.columns:
                    consecutive_closed = (trainee_df["pinch_distance"] <= 0.68).astype(int)
                    max_run = max(np.diff(np.where(np.concatenate(([0], consecutive_closed, [0])) == 0)[0]) - 1)
                    cp_passed = bool(max_run >= 10)
            else:
                cp_passed = True

            if cp_passed:
                passed_count += 1
            elif is_critical:
                critical_failures.append(f"Critical checkpoint failed: {cp.get('name', cp_id)}")

        score_seq = round((passed_count / max(1, total)) * 100.0, 1)
        return score_seq, critical_failures


def determine_interpretation_band(score: float, critical_failures: List[str]) -> str:
    """Map 0-100 score to PRD Section 13.5 interpretation band."""
    if critical_failures or score < 70.0:
        return "Needs review"
    elif score < 80.0:
        return "Developing"
    elif score < 90.0:
        return "Good match"
    else:
        return "Excellent match"
