"""Actionable coaching feedback generator synthesizing plain-language corrections."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from motion_mentor.storage.models import ComponentScores, FeedbackItem

HUMAN_FEATURE_NAMES: Dict[str, str] = {
    "index_mcp": "Index finger base (MCP)",
    "index_pip": "Index finger middle joint (PIP)",
    "index_dip": "Index fingertip joint (DIP)",
    "thumb_mcp": "Thumb base joint (MCP)",
    "thumb_ip": "Thumb tip joint (IP)",
    "thumb_cmc": "Thumb carpal joint (CMC)",
    "middle_pip": "Middle finger joint (PIP)",
    "pinch_distance": "Thumb-to-index pinch aperture",
    "palm_pitch": "Wrist pitch (up/down tilt)",
    "palm_roll": "Wrist roll (inward/outward rotation)",
    "wrist_x": "Horizontal hand position (X)",
    "wrist_y": "Vertical hand position (Y)",
    "wrist_velocity": "Movement speed",
    "wrist_jerk": "Movement smoothness",
}


class FeedbackGenerator:
    """Ranks deviations and generates plain-language coaching recommendations."""

    def generate_feedback(
        self,
        component_scores: ComponentScores,
        per_feature_z: Dict[str, float],
        aligned_ref_df: pd.DataFrame,
        aligned_trainee_df: pd.DataFrame,
        critical_failures: Optional[List[str]] = None,
        max_items: int = 3,
    ) -> List[FeedbackItem]:
        """
        Generate ranked actionable feedback items.
        """
        feedback_items: List[FeedbackItem] = []

        # 1. Critical Failures receive top priority
        if critical_failures:
            for fail in critical_failures:
                feedback_items.append(
                    FeedbackItem(
                        component="Sequence",
                        severity="high",
                        message=fail,
                        time_sec=0.0,
                        measured_deviation="Required checkpoint omitted or incomplete",
                        recommendation="Review the required activity stages and ensure each checkpoint is completed.",
                    )
                )

        # 2. Rank features by Z-score error
        ranked_features = sorted(
            [(feat, z) for feat, z in per_feature_z.items() if z > 0.8],
            key=lambda item: item[1],
            reverse=True,
        )

        for feat, mean_z in ranked_features:
            if len(feedback_items) >= max_items:
                break

            ref_mean_col = f"{feat}_mean"
            if ref_mean_col not in aligned_ref_df.columns or feat not in aligned_trainee_df.columns:
                continue

            # Alignment can hand back object-dtype columns; same cast as scoring.
            ref_vals = np.asarray(aligned_ref_df[ref_mean_col].to_numpy(), dtype=float)
            trainee_vals = np.asarray(aligned_trainee_df[feat].to_numpy(), dtype=float)
            time_ms = aligned_trainee_df["timestamp_ms"].to_numpy()

            # Find peak error step
            abs_errors = np.abs(trainee_vals - ref_vals)
            if not np.isfinite(abs_errors).any():
                continue  # feature undefined on every aligned frame; no measurable deviation
            peak_idx = int(np.nanargmax(abs_errors))
            peak_time_sec = round(float((time_ms[peak_idx] - time_ms[0]) / 1000.0), 1)
            raw_diff = float(trainee_vals[peak_idx] - ref_vals[peak_idx])

            feat_label = HUMAN_FEATURE_NAMES.get(feat, feat.replace("_", " ").title())
            severity = "high" if mean_z >= 2.0 else "medium"

            # Contextual plain-language message
            if "angle" in feat or "mcp" in feat or "pip" in feat or "dip" in feat or "ip" in feat:
                comp = "Pose"
                direction = "more flexed" if raw_diff > 0 else "more extended"
                msg = f"Your {feat_label} was approximately {abs(raw_diff):.0f}° {direction} than the reference at t={peak_time_sec}s."
                rec = f"Adjust your finger angle: keep the joint {'less bent' if raw_diff > 0 else 'more curved'}."
            elif "pinch" in feat:
                comp = "Pose"
                msg = f"Pinch aperture was {'too wide' if raw_diff > 0 else 'closed prematurely'} at t={peak_time_sec}s."
                rec = "Bring your thumb tip and index fingertip together into a firm contact hold."
            elif "pitch" in feat or "roll" in feat or "yaw" in feat:
                comp = "Orientation"
                msg = f"Wrist rotation differed by approximately {abs(raw_diff):.0f}° from the reference at t={peak_time_sec}s."
                rec = "Rotate your wrist earlier during the approach to align palm inward."
            elif "wrist_" in feat:
                comp = "Trajectory"
                msg = f"Hand trajectory deviated from the demonstrated path around t={peak_time_sec}s."
                rec = "Keep your hand centered in the workspace during the movement."
            elif "jerk" in feat or "velocity" in feat:
                comp = "Smoothness"
                msg = f"Motion speed was uneven with sudden hesitation around t={peak_time_sec}s."
                rec = "Perform the reach and pinch in one continuous, steady motion."
            else:
                comp = "Pose"
                msg = f"{feat_label} showed noticeable deviation around t={peak_time_sec}s."
                rec = "Match the reference demonstration posture."

            dev_str = f"Deviation: {abs(raw_diff):.1f} ({mean_z:.1f}σ)"
            feedback_items.append(
                FeedbackItem(
                    component=comp,
                    severity=severity,
                    message=msg,
                    time_sec=peak_time_sec,
                    measured_deviation=dev_str,
                    recommendation=rec,
                )
            )

        # If performance was excellent with no major deviation
        if not feedback_items:
            feedback_items.append(
                FeedbackItem(
                    component="Overall",
                    severity="low",
                    message="Excellent execution across all dimensions.",
                    time_sec=0.0,
                    measured_deviation="Within expert variability tolerance envelope (< 1σ)",
                    recommendation="Maintain current technique and timing.",
                )
            )

        return feedback_items[:max_items]
