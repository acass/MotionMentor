"""Expert Reference Profile Builder combining multiple demonstrations into tolerance envelopes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from motion_mentor.comparison.dtw import align_sequences, constrained_dtw
from motion_mentor.storage.models import Activity, ReferenceProfile, Session, generate_uuid

logger = logging.getLogger(__name__)

# Minimum variability thresholds (\epsilon) to prevent division by zero in Z-score calculation
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


class ReferenceProfileBuilder:
    """Constructs an expert reference profile with mean trajectories and tolerance envelopes."""

    def __init__(
        self,
        activity: Activity,
        alignment_features: Optional[List[str]] = None,
    ) -> None:
        self.activity = activity
        self.alignment_features = alignment_features or CORE_ALIGNMENT_FEATURES

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

        k = len(sessions_and_dfs)
        feature_cols = [
            c for c in sessions_and_dfs[0][1].columns
            if c not in ("frame_index", "timestamp_ms", "valid")
        ]

        # 1. Medoid Selection (if multiple takes)
        if k == 1:
            medoid_idx = 0
            medoid_session, medoid_df = sessions_and_dfs[0]
        else:
            # Pairwise DTW distance matrix
            dist_matrix = np.zeros((k, k))
            align_cols = [c for c in self.alignment_features if c in feature_cols]

            for i in range(k):
                seq_i = sessions_and_dfs[i][1][align_cols].to_numpy()
                for j in range(i + 1, k):
                    seq_j = sessions_and_dfs[j][1][align_cols].to_numpy()
                    d, _, _ = constrained_dtw(seq_i, seq_j)
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

            # Medoid has the minimum total distance to all other demonstrations
            sum_distances = np.sum(dist_matrix, axis=1)
            medoid_idx = int(np.argmin(sum_distances))
            medoid_session, medoid_df = sessions_and_dfs[medoid_idx]

        # 2. Align all takes to the Medoid sequence
        align_cols = [c for c in self.alignment_features if c in feature_cols]
        medoid_align_seq = medoid_df[align_cols].to_numpy()
        n_medoid = len(medoid_df)

        aligned_takes_by_feature: Dict[str, List[np.ndarray]] = {col: [] for col in feature_cols}

        for idx, (sess, df) in enumerate(sessions_and_dfs):
            if idx == medoid_idx:
                for col in feature_cols:
                    aligned_takes_by_feature[col].append(medoid_df[col].to_numpy())
            else:
                seq_curr = df[align_cols].to_numpy()
                _, path, _ = constrained_dtw(medoid_align_seq, seq_curr)

                # Resample this take to medoid grid: for each medoid frame i, average mapped frames j
                mapped_by_medoid: Dict[int, List[int]] = {i: [] for i in range(n_medoid)}
                for i_med, j_curr in path:
                    mapped_by_medoid[i_med].append(j_curr)

                for col in feature_cols:
                    col_vals = df[col].to_numpy()
                    resampled = np.zeros(n_medoid)
                    for i_med in range(n_medoid):
                        j_list = mapped_by_medoid[i_med]
                        if j_list:
                            resampled[i_med] = np.mean(col_vals[j_list])
                        else:
                            resampled[i_med] = col_vals[min(i_med, len(col_vals) - 1)]
                    aligned_takes_by_feature[col].append(resampled)

        # 3. Calculate Mean and Standard Deviation Envelopes
        envelope_data: Dict[str, np.ndarray] = {
            "frame_index": np.arange(n_medoid),
            "timestamp_ms": medoid_df["timestamp_ms"].to_numpy(),
        }

        for col in feature_cols:
            stack = np.vstack(aligned_takes_by_feature[col])  # shape (K, n_medoid)
            mean_vals = np.mean(stack, axis=0)
            if k > 1:
                std_vals = np.std(stack, axis=0)
            else:
                # Single take: apply baseline tolerance
                std_vals = np.zeros(n_medoid)

            # Enforce minimum variability threshold (\epsilon)
            min_tol = get_feature_min_tolerance(col)
            clamped_std = np.maximum(std_vals, min_tol)

            envelope_data[f"{col}_mean"] = mean_vals
            envelope_data[f"{col}_std"] = clamped_std

        envelope_df = pd.DataFrame(envelope_data)

        # 4. Save Reference Profile
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
