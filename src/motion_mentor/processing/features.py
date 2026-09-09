"""Explainable geometric and kinematic feature derivation for hand motion."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from motion_mentor.storage.models import LandmarkFrameRecord

# Standard joint angle definitions: (name, landmark_A, landmark_B_vertex, landmark_C)
JOINT_DEFINITIONS: List[Tuple[str, int, int, int]] = [
    # Thumb
    ("thumb_cmc", 0, 1, 2),
    ("thumb_mcp", 1, 2, 3),
    ("thumb_ip", 2, 3, 4),
    # Index finger
    ("index_mcp", 0, 5, 6),
    ("index_pip", 5, 6, 7),
    ("index_dip", 6, 7, 8),
    # Middle finger
    ("middle_mcp", 0, 9, 10),
    ("middle_pip", 9, 10, 11),
    ("middle_dip", 10, 11, 12),
    # Ring finger
    ("ring_mcp", 0, 13, 14),
    ("ring_pip", 13, 14, 15),
    ("ring_dip", 14, 15, 16),
    # Pinky finger
    ("pinky_mcp", 0, 17, 18),
    ("pinky_pip", 17, 18, 19),
    ("pinky_dip", 18, 19, 20),
]


def compute_3d_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute angle at vertex B formed by segments BA and BC in degrees (0..180).
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return float(np.degrees(angle_rad))


def compute_palm_orientation(landmarks: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Compute palm normal vector and Euler angles (pitch, roll, yaw) in degrees.
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    longitudinal = middle_mcp - wrist
    transverse = pinky_mcp - index_mcp

    normal = np.cross(transverse, longitudinal)
    norm_n = np.linalg.norm(normal)
    if norm_n > 1e-6:
        normal = normal / norm_n
    else:
        normal = np.array([0.0, 0.0, 1.0])

    # Pitch: rotation around horizontal X axis
    pitch = float(np.degrees(np.arctan2(normal[1], normal[2])))
    # Roll: rotation around longitudinal Y axis
    roll = float(np.degrees(np.arctan2(normal[0], normal[2])))
    # Yaw: orientation of longitudinal axis in XY plane
    yaw = float(np.degrees(np.arctan2(longitudinal[0], -longitudinal[1])))

    return normal, pitch, roll, yaw


def extract_frame_features(
    landmarks_3d: np.ndarray,  # shape (21, 3)
    palm_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Extract geometric features (joint angles, distances, palm orientation) from a single frame.
    """
    pts = np.array(landmarks_3d, dtype=np.float64)
    features: Dict[str, float] = {}

    # 1. 14 Joint Angles (degrees)
    for name, idx_a, idx_b, idx_c in JOINT_DEFINITIONS:
        features[name] = compute_3d_angle(pts[idx_a], pts[idx_b], pts[idx_c])

    # 2. Key Pairwise Distances (scale-normalized)
    scale = max(palm_scale, 1e-6)
    # Pinch distance (thumb tip to index tip)
    features["pinch_distance"] = float(np.linalg.norm(pts[4] - pts[8])) / scale
    # Thumb to middle tip distance
    features["thumb_middle_distance"] = float(np.linalg.norm(pts[4] - pts[12])) / scale
    # Hand spread (thumb tip to pinky tip)
    features["hand_spread"] = float(np.linalg.norm(pts[4] - pts[20])) / scale

    # 3. Palm Orientation
    normal, pitch, roll, yaw = compute_palm_orientation(pts)
    features["palm_normal_x"] = float(normal[0])
    features["palm_normal_y"] = float(normal[1])
    features["palm_normal_z"] = float(normal[2])
    features["palm_pitch"] = pitch
    features["palm_roll"] = roll
    features["palm_yaw"] = yaw

    return features


def extract_session_features_df(
    records: List[LandmarkFrameRecord],
    global_trajectory: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Extract full time-series features (geometric + kinematic) for an entire session.
    Returns a Pandas DataFrame with all features indexed by timestamp.
    """
    rows: List[Dict[str, float]] = []

    for i, rec in enumerate(records):
        if not rec.hands or not rec.valid:
            # Missing/invalid frame
            row = {"frame_index": rec.frame_index, "timestamp_ms": rec.timestamp_ms, "valid": False}
            rows.append(row)
            continue

        hand = rec.hands[0]
        pts = np.array(hand.landmarks_image, dtype=np.float64)
        palm_dist = float(np.linalg.norm(pts[9] - pts[0]))

        frame_feats = extract_frame_features(pts, palm_scale=palm_dist)
        frame_feats["frame_index"] = float(rec.frame_index)
        frame_feats["timestamp_ms"] = float(rec.timestamp_ms)

        if global_trajectory is not None and i < len(global_trajectory):
            frame_feats["wrist_x"] = float(global_trajectory[i, 0])
            frame_feats["wrist_y"] = float(global_trajectory[i, 1])
            frame_feats["wrist_z"] = float(global_trajectory[i, 2])
        else:
            frame_feats["wrist_x"] = float(pts[0, 0])
            frame_feats["wrist_y"] = float(pts[0, 1])
            frame_feats["wrist_z"] = float(pts[0, 2])

        frame_feats["valid"] = True
        rows.append(frame_feats)

    df = pd.DataFrame(rows)
    if df.empty or len(df) < 2:
        return df

    # Compute Kinematics (Velocities, Accelerations, Jerk)
    dt_sec = np.maximum(df["timestamp_ms"].diff() / 1000.0, 1e-4)

    # 1. Wrist Linear Velocity
    dx = df["wrist_x"].diff()
    dy = df["wrist_y"].diff()
    dz = df["wrist_z"].diff()
    displacement = np.sqrt(dx**2 + dy**2 + dz**2)
    df["wrist_velocity"] = (displacement / dt_sec).fillna(0.0)

    # 2. Wrist Acceleration
    df["wrist_acceleration"] = (df["wrist_velocity"].diff() / dt_sec).fillna(0.0)

    # 3. Motion Smoothness / Jerk (absolute derivative of acceleration)
    df["wrist_jerk"] = (df["wrist_acceleration"].diff() / dt_sec).abs().fillna(0.0)

    # 4. Pinch Velocity (rate of finger closure/opening)
    if "pinch_distance" in df.columns:
        df["pinch_velocity"] = (df["pinch_distance"].diff() / dt_sec).fillna(0.0)

    return df
