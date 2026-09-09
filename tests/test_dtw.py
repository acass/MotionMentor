"""Unit tests for Constrained Dynamic Time Warping (DTW)."""

import numpy as np
from motion_mentor.comparison.dtw import align_sequences, constrained_dtw


def test_dtw_identity() -> None:
    # 20 frames of a 3D signal
    t = np.linspace(0, 2 * np.pi, 20)
    seq_a = np.column_stack([np.sin(t), np.cos(t), t])

    dist, path, _ = constrained_dtw(seq_a, seq_a, window_ratio=0.25)
    assert np.isclose(dist, 0.0, atol=1e-5)
    # Path should be strictly diagonal
    for i, j in path:
        assert i == j


def test_dtw_time_stretch() -> None:
    # Sequence A: 20 frames
    t_a = np.linspace(0, np.pi, 20)
    seq_a = np.column_stack([np.sin(t_a), np.cos(t_a)])

    # Sequence B: Same motion performed at half speed (40 frames)
    t_b = np.linspace(0, np.pi, 40)
    seq_b = np.column_stack([np.sin(t_b), np.cos(t_b)])

    dist, path, _ = constrained_dtw(seq_a, seq_b, window_ratio=0.5)
    # Distance should be very small since shapes are identical
    assert dist < 0.05
    assert len(path) >= 40
    assert path[0] == (0, 0)
    assert path[-1] == (19, 39)


def test_align_sequences() -> None:
    seq_a = np.array([[0.0], [1.0], [2.0]])
    seq_b = np.array([[0.0], [0.5], [1.0], [2.0]])
    path = [(0, 0), (0, 1), (1, 2), (2, 3)]

    aligned_a, aligned_b = align_sequences(seq_a, seq_b, path)
    assert aligned_a.shape == (4, 1)
    assert aligned_b.shape == (4, 1)
    assert aligned_a[1, 0] == 0.0
    assert aligned_b[1, 0] == 0.5
