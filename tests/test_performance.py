"""Performance budget tests for the hot numeric paths.

Budgets are generous (several x the measured cost on an Apple Silicon dev box)
so they catch algorithmic regressions, not scheduler jitter.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from motion_mentor.comparison.dtw import constrained_dtw
from motion_mentor.processing.smoothing import OneEuroFilter


def _timed(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.parametrize("n", [150, 600])
def test_dtw_budget(n: int) -> None:
    """A 20 s take at 30 FPS is 600 frames; DTW must stay well under a second."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n, 12))
    b = rng.normal(size=(n, 12))
    elapsed = _timed(lambda: constrained_dtw(a, b, window_ratio=0.15))
    budget = 0.5 if n == 150 else 3.0
    assert elapsed < budget, f"DTW n={n} took {elapsed:.3f}s (budget {budget}s)"


def test_dtw_scales_subquadratically_with_band() -> None:
    """Sakoe-Chiba band must actually bound work: 2x length is not 4x time."""
    rng = np.random.default_rng(1)
    small = _timed(lambda: constrained_dtw(rng.normal(size=(300, 8)), rng.normal(size=(300, 8)), window_ratio=0.1))
    large = _timed(lambda: constrained_dtw(rng.normal(size=(600, 8)), rng.normal(size=(600, 8)), window_ratio=0.05))
    assert large < small * 3.5, f"band not bounding work: {small:.3f}s vs {large:.3f}s"


def test_smoother_budget() -> None:
    """One Euro smoothing of 600 frames x 21 landmarks x 3 axes under 1 s."""
    smoother = OneEuroFilter()
    rng = np.random.default_rng(2)
    frames = rng.normal(size=(600, 63))

    def run() -> None:
        for i, f in enumerate(frames):
            smoother.filter(f, i / 30.0)

    elapsed = _timed(run, repeats=1)
    assert elapsed < 1.0, f"smoothing took {elapsed:.3f}s"
