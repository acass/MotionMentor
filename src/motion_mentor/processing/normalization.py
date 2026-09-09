"""Coordinate normalization and canonical hand-frame transformation."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from motion_mentor.storage.models import HandLandmarkData, LandmarkFrameRecord


def normalize_hand_landmarks(
    landmarks_3d: np.ndarray,  # shape (21, 3)
    align_rotation: bool = True,
    mirror_x: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Normalize 21 3D hand landmarks into a canonical scale- and orientation-invariant frame.

    Args:
        landmarks_3d: (21, 3) array of [x, y, z] points.
        align_rotation: If True, rotate into palm orthonormal coordinate basis.
        mirror_x: If True, mirror X axis (for left-hand normalization).

    Returns:
        (canonical_landmarks, palm_scale, rotation_matrix):
            - canonical_landmarks: (21, 3) normalized points (wrist at (0,0,0), palm length = 1.0)
            - palm_scale: Original Euclidean distance between wrist and middle MCP
            - rotation_matrix: 3x3 rotation matrix used to align to canonical frame
    """
    pts = np.array(landmarks_3d, dtype=np.float64)

    if mirror_x:
        pts[:, 0] = -pts[:, 0]

    # 1. Translation: Wrist (0) becomes local origin (0, 0, 0)
    wrist = pts[0].copy()
    translated = pts - wrist

    # 2. Scale invariance: Normalize by wrist (0) to middle MCP (9) distance
    middle_mcp = translated[9]
    palm_scale = float(np.linalg.norm(middle_mcp))
    epsilon = 1e-6
    scale_factor = max(palm_scale, epsilon)
    scaled = translated / scale_factor

    if not align_rotation:
        return scaled.astype(np.float32), palm_scale, np.eye(3, dtype=np.float32)

    # 3. Canonical Orthonormal Basis
    # Y-axis: Along the hand longitudinal axis (wrist -> middle MCP)
    vec_y = scaled[9]
    norm_y = np.linalg.norm(vec_y)
    if norm_y < epsilon:
        u_y = np.array([0.0, 1.0, 0.0])
    else:
        u_y = vec_y / norm_y

    # Transverse vector across palm base: index MCP (5) -> pinky MCP (17)
    vec_transverse = scaled[17] - scaled[5]

    # Z-axis (Palm Normal): cross product of transverse and longitudinal
    # For a right hand with palm facing user, cross(u_y, transverse) points out of palm
    cross_z = np.cross(vec_transverse, u_y)
    norm_z = np.linalg.norm(cross_z)
    if norm_z < epsilon:
        u_z = np.array([0.0, 0.0, 1.0])
    else:
        u_z = cross_z / norm_z

    # X-axis: Orthonormal completion
    u_x = np.cross(u_y, u_z)
    norm_x = np.linalg.norm(u_x)
    if norm_x > epsilon:
        u_x = u_x / norm_x

    # Basis matrix: Rows are canonical axes
    rot_matrix = np.vstack([u_x, u_y, u_z])  # shape (3, 3)

    # Rotate all points into canonical basis: P_canonical = scaled @ rot_matrix.T
    canonical = scaled @ rot_matrix.T

    return canonical.astype(np.float32), palm_scale, rot_matrix.astype(np.float32)


def extract_global_trajectory(
    records: List[LandmarkFrameRecord],
) -> np.ndarray:
    """
    Extract global wrist trajectory (landmark 0) across all frames.
    Returns:
        (N, 3) array of wrist positions [x, y, z] over time.
    """
    wrist_positions: List[List[float]] = []
    for rec in records:
        if rec.hands and rec.valid and len(rec.hands[0].landmarks_image) > 0:
            wrist_positions.append(rec.hands[0].landmarks_image[0])
        else:
            wrist_positions.append([np.nan, np.nan, np.nan])
    return np.array(wrist_positions, dtype=np.float32)


from motion_mentor.processing.features import compute_palm_orientation


def extract_global_orientations(
    records: List[LandmarkFrameRecord],
) -> np.ndarray:
    """
    Extract global palm orientation [normal_x, normal_y, normal_z, pitch, roll, yaw]
    across all frames prior to canonical rotation.
    Returns:
        (N, 6) array
    """
    orientations: List[List[float]] = []
    for rec in records:
        if rec.hands and rec.valid and len(rec.hands[0].landmarks_image) >= 21:
            pts = np.array(rec.hands[0].landmarks_image, dtype=np.float64)
            normal, pitch, roll, yaw = compute_palm_orientation(pts)
            orientations.append([float(normal[0]), float(normal[1]), float(normal[2]), pitch, roll, yaw])
        else:
            orientations.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return np.array(orientations, dtype=np.float32)


def normalize_session_records(
    records: List[LandmarkFrameRecord],
    mirror_left_hand: bool = False,
) -> Tuple[List[LandmarkFrameRecord], np.ndarray, np.ndarray]:
    """
    Normalize an entire landmark sequence:
    Returns:
        (normalized_records, global_wrist_trajectory, global_palm_orientations)
    """
    global_wrist = extract_global_trajectory(records)
    global_orientations = extract_global_orientations(records)
    normalized_records: List[LandmarkFrameRecord] = []

    for rec in records:
        if not rec.hands or not rec.valid:
            normalized_records.append(rec)
            continue

        hand = rec.hands[0]
        pts = np.array(hand.landmarks_image, dtype=np.float32)
        should_mirror = mirror_left_hand and (hand.handedness == "Left")

        canonical, scale, _ = normalize_hand_landmarks(
            pts,
            align_rotation=True,
            mirror_x=should_mirror,
        )

        norm_hand = HandLandmarkData(
            hand_track_id=hand.hand_track_id,
            handedness=hand.handedness,
            handedness_score=hand.handedness_score,
            landmarks_image=canonical.tolist(),
            landmarks_world=hand.landmarks_world,
            valid=hand.valid,
        )
        norm_rec = LandmarkFrameRecord(
            session_id=rec.session_id,
            frame_index=rec.frame_index,
            timestamp_ms=rec.timestamp_ms,
            hands=[norm_hand],
            valid=rec.valid,
        )
        normalized_records.append(norm_rec)

    return normalized_records, global_wrist, global_orientations

