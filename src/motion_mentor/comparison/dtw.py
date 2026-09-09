"""Constrained Dynamic Time Warping (DTW) with Sakoe-Chiba window constraints."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


def compute_frame_distance(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute weighted Euclidean distance between two feature vectors."""
    diff = vec_a - vec_b
    if weights is not None:
        return float(np.sqrt(np.sum(weights * (diff ** 2))))
    return float(np.linalg.norm(diff))


def constrained_dtw(
    seq_a: np.ndarray,  # shape (N, D)
    seq_b: np.ndarray,  # shape (M, D)
    window_ratio: float = 0.25,
    feature_weights: Optional[np.ndarray] = None,
) -> Tuple[float, List[Tuple[int, int]], np.ndarray]:
    """
    Constrained Dynamic Time Warping using Sakoe-Chiba band.

    Args:
        seq_a: Reference sequence of shape (N, D).
        seq_b: Trainee sequence of shape (M, D).
        window_ratio: Constraint band width as fraction of max sequence length.
        feature_weights: Optional 1D weights of length D.

    Returns:
        (normalized_distance, warping_path, cost_matrix):
            - normalized_distance: Total accumulated cost divided by path length.
            - warping_path: List of index pairs [(i_0, j_0), (i_1, j_1), ..., (N-1, M-1)].
            - cost_matrix: (N, M) cumulative cost matrix.
    """
    n, d = seq_a.shape
    m = seq_b.shape[0]

    if n == 0 or m == 0:
        raise ValueError("Sequences must not be empty.")

    # Calculate Sakoe-Chiba window width
    window_w = max(5, int(max(n, m) * window_ratio))

    # Normalize feature weights if provided
    weights = None
    if feature_weights is not None:
        w_arr = np.array(feature_weights, dtype=np.float64)
        weights = w_arr / np.sum(w_arr)

    # Initialize cumulative cost matrix with infinity
    cost = np.full((n, m), np.inf, dtype=np.float64)

    # Base cost at (0, 0)
    cost[0, 0] = compute_frame_distance(seq_a[0], seq_b[0], weights)

    # Populate cost matrix within Sakoe-Chiba band
    for i in range(n):
        # Window bounds around diagonal
        j_min = max(0, i - window_w)
        j_max = min(m, i + window_w + 1)

        for j in range(j_min, j_max):
            if i == 0 and j == 0:
                continue

            step_dist = compute_frame_distance(seq_a[i], seq_b[j], weights)

            choices = []
            if i > 0:
                choices.append(cost[i - 1, j])
            if j > 0:
                choices.append(cost[i, j - 1])
            if i > 0 and j > 0:
                choices.append(cost[i - 1, j - 1])

            if choices:
                min_prior = min(choices)
                if not np.isinf(min_prior):
                    cost[i, j] = step_dist + min_prior

    # Fallback if window was too narrow to reach (n-1, m-1)
    if np.isinf(cost[n - 1, m - 1]):
        # Widen window and re-run unconditionally without band
        return constrained_dtw(seq_a, seq_b, window_ratio=1.0, feature_weights=feature_weights)

    # Backtrack to find the optimal warping path
    i = n - 1
    j = m - 1
    path: List[Tuple[int, int]] = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            diag = cost[i - 1, j - 1]
            up = cost[i - 1, j]
            left = cost[i, j - 1]

            min_val = min(diag, up, left)
            if min_val == diag:
                i -= 1
                j -= 1
            elif min_val == up:
                i -= 1
            else:
                j -= 1

        path.append((i, j))

    path.reverse()
    norm_distance = float(cost[n - 1, m - 1] / len(path))

    return norm_distance, path, cost


def align_sequences(
    ref_seq: np.ndarray,      # shape (N, D)
    trainee_seq: np.ndarray,  # shape (M, D)
    warping_path: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample sequences along the warping path so each step k has matched reference and trainee frames.

    Returns:
        (aligned_ref, aligned_trainee): Both arrays of shape (K, D) where K = len(warping_path).
    """
    ref_indices = [p[0] for p in warping_path]
    trainee_indices = [p[1] for p in warping_path]

    aligned_ref = ref_seq[ref_indices]
    aligned_trainee = trainee_seq[trainee_indices]

    return aligned_ref, aligned_trainee
