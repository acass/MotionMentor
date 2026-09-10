"""Expert Reference Profile Builder combining multiple demonstrations into tolerance envelopes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from motion_mentor.comparison.dtw import constrained_dtw
from motion_mentor.storage.models import Activity, ReferenceProfile, Session, generate_uuid

logger = logging.getLogger(__name__)

# Minimum variability thresholds (\epsilon) to prevent division by zero in Z-score calculation.
# These represent the tracker's own jitter, not expert skill: below this spread we cannot
# distinguish a real difference from MediaPipe noise.
# ponytail: guessed constants. Replace with measured values from a static-hold recording
# (scripts/measure_noise_floor.py) once one exists for the production camera rig.
MIN_FEATURE_TOLERANCES: Dict[str, float] = {
    # Joint angles: minimum 4.0 degrees standard deviation
    "angle_default": 4.0,
    # Distances (pinch aperture, spread): minimum 0.04 normalized units
    "distance_default": 0.04,
    # Positions (wrist x, y, z): minimum 0.03 normalized units
    "position_default": 0.03,
    # Palm normal vector: minimum 0.05
    "normal_default": 0.05,
    # Velocities: minimum 0.1
    "velocity_default": 0.1,
}

CORE_ALIGNMENT_FEATURES = [
    "index_mcp", "index_pip", "thumb_mcp", "thumb_ip",
    "pinch_distance", "palm_pitch", "palm_roll", "wrist_x", "wrist_y",
]

# A take must have at least this fraction of frames tracked to enter an envelope.
MIN_TAKE_COVERAGE_PCT = 90.0

# An envelope frame needs this fraction of takes reporting a real value before its
# mean and std mean anything. Below it the frame is left NaN and scoring skips it.
MIN_FRAME_SUPPORT_RATIO = 0.6

# Takes whose total DTW distance to the others exceeds median + K * MAD are outliers.
OUTLIER_MAD_K = 3.0


def get_feature_min_tolerance(feature_name: str) -> float:
    """Return minimum standard deviation tolerance epsilon for a feature."""
    if "velocity" in feature_name or "acceleration" in feature_name or "jerk" in feature_name:
        return MIN_FEATURE_TOLERANCES["velocity_default"]
    elif "distance" in feature_name or "spread" in feature_name:
        return MIN_FEATURE_TOLERANCES["distance_default"]
    elif feature_name in ("wrist_x", "wrist_y", "wrist_z"):
        return MIN_FEATURE_TOLERANCES["position_default"]
    elif "normal" in feature_name:
        return MIN_FEATURE_TOLERANCES["normal_default"]
    else:
        # Joint angles and pitch/roll/yaw
        return MIN_FEATURE_TOLERANCES["angle_default"]


def alignment_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """
    Build a gap-free matrix for DTW alignment only.

    Untracked frames arrive as NaN. DTW compares every frame against every other, so a
    single NaN would poison whole rows of the cost matrix. Alignment only needs a
    continuous curve to follow, so gaps are bridged here. The bridged values are never
    used for statistics or scoring: those read the original NaN-bearing columns.
    """
    filled = df[cols].interpolate(limit_direction="both")
    # A column that is NaN end to end cannot be interpolated; it contributes nothing.
    return filled.fillna(0.0).to_numpy()


def take_coverage_pct(df: pd.DataFrame) -> float:
    """Fraction of frames in a take with a tracked hand, as a percentage."""
    if "valid" in df.columns:
        return float(df["valid"].mean() * 100.0)
    if len(df) == 0:
        return 0.0
    # No explicit validity column: infer from the first feature column.
    feature_cols = [c for c in df.columns if c not in ("frame_index", "timestamp_ms")]
    if not feature_cols:
        return 100.0
    return float(df[feature_cols[0]].notna().mean() * 100.0)


class ReferenceProfileBuilder:
    """Constructs an expert reference profile with mean trajectories and tolerance envelopes."""

    def __init__(
        self,
        activity: Activity,
        alignment_features: Optional[List[str]] = None,
    ) -> None:
        self.activity = activity
        self.alignment_features = alignment_features or CORE_ALIGNMENT_FEATURES

    def _reject_takes(
        self,
        sessions_and_dfs: List[Tuple[Session, pd.DataFrame]],
        align_cols: List[str],
    ) -> Tuple[List[Tuple[Session, pd.DataFrame]], np.ndarray]:
        """
        Drop takes that should not shape the envelope, and return the survivors plus the
        pairwise DTW distance matrix over them.

        Two gates. First a quality gate: a poorly tracked take contributes dropout, not
        expert variability. Then an outlier gate: one botched demonstration averaged in
        with full weight widens the tolerance envelope permanently, and every trainee
        scored afterwards benefits from a mistake nobody noticed.
        """
        original_k = len(sessions_and_dfs)

        kept: List[Tuple[Session, pd.DataFrame]] = []
        for sess, df in sessions_and_dfs:
            coverage = take_coverage_pct(df)
            if original_k > 1 and coverage < MIN_TAKE_COVERAGE_PCT:
                logger.warning(
                    "Excluding take %s from reference: detection coverage %.1f%% below %.1f%%",
                    sess.session_id, coverage, MIN_TAKE_COVERAGE_PCT,
                )
                continue
            kept.append((sess, df))

        k = len(kept)
        dist_matrix = np.zeros((k, k))

        # Below three takes there is nothing to say about which one is unusual, so the
        # outlier gate is skipped. The rejection guard below still applies.
        if k > 2:
            sequences = [alignment_matrix(df, align_cols) for _, df in kept]
            for i in range(k):
                for j in range(i + 1, k):
                    d, _, _ = constrained_dtw(sequences[i], sequences[j])
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

            sum_distances = np.sum(dist_matrix, axis=1)
            median = float(np.median(sum_distances))
            # Median absolute deviation, not standard deviation: the thing being detected
            # is precisely the outlier that would inflate a standard deviation and so hide
            # itself.
            mad = float(np.median(np.abs(sum_distances - median)))
            if mad > 0.0:
                threshold = median + OUTLIER_MAD_K * mad
                survivor_idx = [i for i in range(k) if sum_distances[i] <= threshold]
                for i in range(k):
                    if i not in survivor_idx:
                        logger.warning(
                            "Excluding take %s from reference: DTW distance %.4f exceeds "
                            "outlier threshold %.4f (median %.4f, MAD %.4f)",
                            kept[i][0].session_id, sum_distances[i], threshold, median, mad,
                        )
                if len(survivor_idx) < k:
                    kept = [kept[i] for i in survivor_idx]
                    dist_matrix = dist_matrix[np.ix_(survivor_idx, survivor_idx)]

        if original_k > 1 and len(kept) * 2 < original_k:
            raise ValueError(
                f"Reference build rejected {original_k - len(kept)} of {original_k} takes. "
                "That is a bad recording session, not a few outliers. Inspect the takes "
                "and re-record rather than building an envelope from the remainder."
            )
        if not kept:
            raise ValueError("No expert takes survived quality and outlier rejection.")

        return kept, dist_matrix

    def build_profile(
        self,
        sessions_and_dfs: List[Tuple[Session, pd.DataFrame]],
        output_dir: str | Path = "data/references",
    ) -> Tuple[ReferenceProfile, pd.DataFrame]:
        """
        Build an expert reference from one or more demonstration DataFrames.

        Returns:
            (reference_metadata, reference_envelope_df)
        """
        if not sessions_and_dfs:
            raise ValueError("At least one expert demonstration is required to build a reference.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_cols = [
            c for c in sessions_and_dfs[0][1].columns
            if c not in ("frame_index", "timestamp_ms", "valid")
        ]
        align_cols = [c for c in self.alignment_features if c in feature_cols]

        # 1. Quality and outlier rejection, which also yields the pairwise distances
        sessions_and_dfs, dist_matrix = self._reject_takes(sessions_and_dfs, align_cols)
        k = len(sessions_and_dfs)

        # 2. Medoid selection: the take with the least total distance to all the others
        if k <= 2:
            medoid_idx = 0
        else:
            medoid_idx = int(np.argmin(np.sum(dist_matrix, axis=1)))
        medoid_session, medoid_df = sessions_and_dfs[medoid_idx]

        # 3. Align all takes to the medoid sequence
        medoid_align_seq = alignment_matrix(medoid_df, align_cols)
        n_medoid = len(medoid_df)

        aligned_takes_by_feature: Dict[str, List[np.ndarray]] = {col: [] for col in feature_cols}

        for idx, (sess, df) in enumerate(sessions_and_dfs):
            if idx == medoid_idx:
                for col in feature_cols:
                    aligned_takes_by_feature[col].append(medoid_df[col].to_numpy(dtype=float))
                continue

            seq_curr = alignment_matrix(df, align_cols)
            _, path, _ = constrained_dtw(medoid_align_seq, seq_curr)

            # Resample this take to medoid grid: for each medoid frame i, average mapped frames j
            mapped_by_medoid: Dict[int, List[int]] = {i: [] for i in range(n_medoid)}
            for i_med, j_curr in path:
                mapped_by_medoid[i_med].append(j_curr)

            for col in feature_cols:
                col_vals = df[col].to_numpy(dtype=float)
                resampled = np.full(n_medoid, np.nan)
                for i_med in range(n_medoid):
                    j_list = mapped_by_medoid[i_med]
                    if j_list:
                        window = col_vals[j_list]
                        if np.isfinite(window).any():
                            resampled[i_med] = np.nanmean(window)
                        # else: every mapped frame was untracked, so this take says nothing
                        # about this medoid frame. NaN is the honest answer.
                    else:
                        # No frame of this take mapped here; carry the nearest one.
                        resampled[i_med] = col_vals[min(i_med, len(col_vals) - 1)]
                aligned_takes_by_feature[col].append(resampled)

        # 4. Mean and standard deviation envelopes, ignoring untracked frames
        envelope_data: Dict[str, np.ndarray] = {
            "frame_index": np.arange(n_medoid),
            "timestamp_ms": medoid_df["timestamp_ms"].to_numpy(),
        }

        min_support = max(1, int(np.ceil(MIN_FRAME_SUPPORT_RATIO * k)))
        support_total = np.zeros(n_medoid)

        for col in feature_cols:
            stack = np.vstack(aligned_takes_by_feature[col])  # shape (K, n_medoid)
            support = np.isfinite(stack).sum(axis=0)
            support_total += support
            usable = support >= min_support

            mean_vals = np.full(n_medoid, np.nan)
            std_vals = np.full(n_medoid, np.nan)
            if usable.any():
                # nanmean/nanstd warn on all-NaN columns, so only touch usable frames.
                mean_vals[usable] = np.nanmean(stack[:, usable], axis=0)
                if k > 1:
                    std_vals[usable] = np.nanstd(stack[:, usable], axis=0)
                else:
                    # A single take has no spread to measure; the floor is the whole envelope.
                    std_vals[usable] = 0.0

            # Enforce minimum variability threshold (\epsilon), preserving NaN frames
            min_tol = get_feature_min_tolerance(col)
            clamped_std = np.where(usable, np.maximum(std_vals, min_tol), np.nan)

            envelope_data[f"{col}_mean"] = mean_vals
            envelope_data[f"{col}_std"] = clamped_std

        # Average number of takes backing each frame, so consumers can see thin regions.
        envelope_data["support"] = support_total / max(1, len(feature_cols))

        envelope_df = pd.DataFrame(envelope_data)

        unsupported = int((envelope_df["support"] < min_support).sum())
        if unsupported:
            logger.warning(
                "Reference envelope has %d of %d frames below minimum support (%d of %d takes); "
                "those frames are NaN and will be skipped during scoring.",
                unsupported, n_medoid, min_support, k,
            )

        # 5. Save Reference Profile
        ref_id = generate_uuid()
        parquet_path = output_dir / f"{ref_id}_reference.parquet"
        envelope_df.to_parquet(parquet_path, index=False)

        mean_dur = float(np.mean([s.duration_seconds for s, _ in sessions_and_dfs]))

        profile = ReferenceProfile(
            reference_id=ref_id,
            activity_id=self.activity.activity_id,
            activity_name=self.activity.name,
            activity_version=self.activity.version,
            version=1,
            expert_session_ids=[s.session_id for s, _ in sessions_and_dfs],
            medoid_session_id=medoid_session.session_id,
            total_demonstrations=k,
            duration_mean_sec=round(mean_dur, 2),
            profile_path=str(parquet_path),
        )

        return profile, envelope_df
