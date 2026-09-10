"""Untracked frames and botched takes must not corrupt the tolerance envelope."""

import numpy as np
import pandas as pd
import pytest

from motion_mentor.comparison.reference import (
    MIN_FRAME_SUPPORT_RATIO,
    ReferenceProfileBuilder,
)
from motion_mentor.comparison.scoring import compute_component_zscore
from motion_mentor.storage.models import Activity, Session

from test_reference import create_synthetic_expert_df


@pytest.fixture
def activity() -> Activity:
    return Activity(activity_id="reach_and_pinch", name="Reach and Pinch", description="Benchmark")


def take(session_id: str, num_frames: int = 60, rot_offset: float = 0.0):
    df = create_synthetic_expert_df(num_frames=num_frames, rot_offset=rot_offset)
    sess = Session(
        session_id=session_id,
        activity_id="reach_and_pinch",
        user_id="expert_1",
        role="expert",
        duration_seconds=2.0,
    )
    return sess, df


def blank_frames(df: pd.DataFrame, start: int, count: int) -> pd.DataFrame:
    """Mimic a tracking dropout: feature columns go NaN, validity goes false."""
    out = df.copy()
    feature_cols = [c for c in out.columns if c not in ("frame_index", "timestamp_ms", "valid")]
    out.loc[start:start + count - 1, feature_cols] = np.nan
    out.loc[start:start + count - 1, "valid"] = 0
    return out


def test_dropout_in_one_take_does_not_poison_the_envelope(activity, tmp_path):
    takes = [take(f"sess_{i}", rot_offset=float(i)) for i in range(5)]
    # One take loses tracking mid-motion. Every other take still saw those frames.
    takes[2] = (takes[2][0], blank_frames(takes[2][1], 20, 5))

    profile, envelope = ReferenceProfileBuilder(activity).build_profile(takes, output_dir=tmp_path)

    assert profile.total_demonstrations == 5
    mean_cols = [c for c in envelope.columns if c.endswith("_mean")]
    assert envelope[mean_cols].notna().all().all(), "a single take's dropout blanked the envelope"
    assert (envelope["support"] >= MIN_FRAME_SUPPORT_RATIO * 5).all()


def test_frames_no_take_saw_are_left_nan(activity, tmp_path):
    # Every take drops the same window, so nothing is known about it.
    takes = [take(f"sess_{i}", rot_offset=float(i)) for i in range(4)]
    takes = [(s, blank_frames(df, 30, 6)) for s, df in takes]

    _, envelope = ReferenceProfileBuilder(activity).build_profile(takes, output_dir=tmp_path)

    unsupported = envelope["index_mcp_mean"].isna()
    assert unsupported.any(), "unsupported frames should stay NaN rather than be invented"
    # And a NaN envelope contributes nothing to a score instead of producing NaN.
    trainee = np.full(len(envelope), 150.0)
    z = compute_component_zscore(
        trainee, envelope["index_mcp_mean"].to_numpy(), envelope["index_mcp_std"].to_numpy()
    )
    assert np.isfinite(z)


def test_botched_take_is_excluded(activity, tmp_path):
    takes = [take(f"sess_{i}", rot_offset=float(i)) for i in range(6)]
    # A take that wandered off: wildly different motion, still perfectly tracked.
    bad_sess, bad_df = take("sess_botched")
    bad_df = bad_df.copy()
    bad_df["palm_pitch"] = bad_df["palm_pitch"] * -4.0 + 300.0
    bad_df["wrist_x"] = bad_df["wrist_x"] + 3.0
    bad_df["index_mcp"] = 20.0
    takes.append((bad_sess, bad_df))

    profile, _ = ReferenceProfileBuilder(activity).build_profile(takes, output_dir=tmp_path)

    assert "sess_botched" not in profile.expert_session_ids
    assert profile.total_demonstrations == 6


def test_low_coverage_take_is_excluded(activity, tmp_path):
    takes = [take(f"sess_{i}", rot_offset=float(i)) for i in range(5)]
    thin_sess, thin_df = take("sess_thin")
    takes.append((thin_sess, blank_frames(thin_df, 0, 30)))  # 50% coverage

    profile, _ = ReferenceProfileBuilder(activity).build_profile(takes, output_dir=tmp_path)

    assert "sess_thin" not in profile.expert_session_ids


def test_mass_rejection_is_an_error_not_a_thin_envelope(activity, tmp_path):
    # A whole bad session: only two takes are usable out of seven.
    takes = [take(f"sess_{i}", rot_offset=float(i)) for i in range(2)]
    for i in range(5):
        sess, df = take(f"sess_bad_{i}")
        takes.append((sess, blank_frames(df, 0, 40)))

    with pytest.raises(ValueError, match="bad recording session"):
        ReferenceProfileBuilder(activity).build_profile(takes, output_dir=tmp_path)
