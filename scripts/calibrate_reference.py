"""Find how many expert takes are enough, and calibrate the score bands against them.

Two analyses, both read-only. Nothing is written to data/ or to the database.

Sweep
    Rebuild the tolerance envelope from 5, 10, 15 ... takes and report mean standard
    deviation per feature category at each size. Where the standard deviation stops
    growing, the envelope has seen the expert's whole range and more takes add nothing.

Leave-one-out
    For each take, build the envelope from all the others and score the held-out take
    against it. That distribution is the expert's own ceiling: what a genuinely correct
    performance scores. Score bands should be set relative to it.

    Read the histogram, not just the mean. A wide or two-humped distribution means the
    expert has more than one valid way of doing this motion, and a single mean-plus-
    standard-deviation envelope is averaging them into a middle that nobody performs.

Usage:
    uv run python scripts/calibrate_reference.py --activity reach_and_pinch
    uv run python scripts/calibrate_reference.py --activity reach_and_pinch --plot out.png
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motion_mentor.app import MotionMentorApp  # noqa: E402
from motion_mentor.comparison.pipeline import (  # noqa: E402
    prepare_session_features,
    score_attempt_against_envelope,
)
from motion_mentor.comparison.reference import (  # noqa: E402
    MIN_FEATURE_TOLERANCES,
    ReferenceProfileBuilder,
    get_feature_min_tolerance,
)
from motion_mentor.comparison.scoring import determine_interpretation_band  # noqa: E402

CATEGORY_LABEL = {
    "angle_default": "angles",
    "distance_default": "distances",
    "position_default": "positions",
    "normal_default": "normals",
    "velocity_default": "velocities",
}


def category_of(feature: str) -> str:
    tolerance = get_feature_min_tolerance(feature)
    for key, value in MIN_FEATURE_TOLERANCES.items():
        if value == tolerance:
            return key
    return "angle_default"


def envelope_spread(envelope) -> dict[str, float]:
    """Mean standard deviation per feature category, ignoring unsupported frames."""
    by_category: dict[str, list[float]] = {key: [] for key in MIN_FEATURE_TOLERANCES}
    for col in envelope.columns:
        if not col.endswith("_std"):
            continue
        feature = col[: -len("_std")]
        values = envelope[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            by_category[category_of(feature)].append(float(np.mean(values)))
    return {k: float(np.mean(v)) for k, v in by_category.items() if v}


def load_takes(app: MotionMentorApp, activity_id: str):
    sessions = app.db.list_sessions(activity_id=activity_id, role="expert")
    takes = []
    for sess in sessions:
        try:
            takes.append((sess, prepare_session_features(app, sess)))
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  skipping {sess.session_id}: {exc}")
    return takes


def run_sweep(activity, takes, out_dir: Path, step: int) -> list[tuple[int, dict[str, float]]]:
    print(f"\n{'=' * 62}\nENVELOPE SWEEP\n{'=' * 62}")
    sizes = [n for n in range(step, len(takes) + 1, step)]
    if len(takes) not in sizes:
        sizes.append(len(takes))

    results = []
    for n in sizes:
        try:
            _, envelope = ReferenceProfileBuilder(activity).build_profile(takes[:n], output_dir=out_dir)
        except ValueError as exc:
            print(f"N={n:<4} build refused: {exc}")
            continue
        spread = envelope_spread(envelope)
        results.append((n, spread))
        parts = " ".join(f"{CATEGORY_LABEL[k]}={v:.3f}" for k, v in spread.items())
        print(f"N={n:<4} {parts}")

    if len(results) >= 2:
        first, last = results[0][1], results[-1][1]
        print("\nGrowth from smallest to largest sample:")
        for key in last:
            if key in first and first[key] > 0:
                change = (last[key] - first[key]) / first[key] * 100.0
                print(f"  {CATEGORY_LABEL[key]:<12}{change:+7.1f}%")
        print("\nA category still growing at the largest N has not saturated: record more takes.")
    return results


def run_leave_one_out(activity, takes, out_dir: Path) -> list[tuple[str, float]]:
    print(f"\n{'=' * 62}\nLEAVE-ONE-OUT\n{'=' * 62}")
    if len(takes) < 3:
        print(f"Only {len(takes)} takes; leave-one-out needs at least 3.")
        return []

    scores: list[tuple[str, float]] = []
    for i, (held_session, held_df) in enumerate(takes):
        rest = [t for j, t in enumerate(takes) if j != i]
        try:
            profile, envelope = ReferenceProfileBuilder(activity).build_profile(rest, output_dir=out_dir)
        except ValueError as exc:
            print(f"  {held_session.session_id[:8]}: envelope build refused: {exc}")
            continue

        result = score_attempt_against_envelope(
            activity=activity,
            ref_df=envelope,
            trainee_df=held_df,
            trainee_duration_sec=held_session.duration_seconds,
            reference_duration_sec=profile.duration_mean_sec,
            trainee_quality=held_session.quality_summary,
        )
        scores.append((held_session.session_id, result.overall_score))
        flags = ", ".join(result.critical_failures) if result.critical_failures else ""
        print(
            f"  {held_session.session_id[:8]}  score {result.overall_score:5.1f}  "
            f"scored frames {result.scored_fraction * 100:5.1f}%  {flags}"
        )

    if not scores:
        return scores

    values = [s for _, s in scores]
    print(f"\n  n         {len(values)}")
    print(f"  mean      {statistics.mean(values):.1f}")
    print(f"  median    {statistics.median(values):.1f}")
    if len(values) > 1:
        print(f"  std dev   {statistics.stdev(values):.1f}")
    print(f"  range     {min(values):.1f} to {max(values):.1f}")

    print("\n  Distribution:")
    for low in range(0, 100, 10):
        count = sum(1 for v in values if low <= v < low + 10)
        if count:
            print(f"    {low:3d}-{low + 9:3d}  {'#' * count} ({count})")

    bands: dict[str, int] = {}
    for _, score in scores:
        band = determine_interpretation_band(score, [])
        bands[band] = bands.get(band, 0) + 1
    print("\n  Bands a held-out expert take currently lands in:")
    for band, count in sorted(bands.items(), key=lambda kv: -kv[1]):
        print(f"    {band:<18}{count}")

    print(
        "\n  Aim for held-out expert takes in the low-to-mid 90s and tightly grouped.\n"
        "  Much lower means the envelope is too tight, or a critical checkpoint is\n"
        "  misfiring. Much higher, or very wide, means the envelope is too loose to\n"
        "  discriminate. Adjust decay_factor in comparison/scoring.py and re-run."
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activity", default="reach_and_pinch", help="Activity name or id")
    parser.add_argument("--config", default=None, help="Config file passed to MotionMentorApp")
    parser.add_argument("--step", type=int, default=5, help="Sweep increment (default 5)")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-loo", action="store_true")
    parser.add_argument("--plot", type=Path, default=None, help="Write the sweep and LOO plot here")
    args = parser.parse_args()

    app = MotionMentorApp(args.config) if args.config else MotionMentorApp()
    activity = app.load_or_create_activity(args.activity)

    print(f"Loading expert takes for '{activity.name}' ({activity.activity_id})...")
    takes = load_takes(app, activity.activity_id)
    print(f"Loaded {len(takes)} expert take(s).")
    if not takes:
        sys.exit("No expert sessions found. Record some first.")

    # Envelopes built here are throwaways; keep them out of data/references.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        sweep = [] if args.skip_sweep else run_sweep(activity, takes, out_dir, args.step)
        loo = [] if args.skip_loo else run_leave_one_out(activity, takes, out_dir)

    if args.plot and (sweep or loo):
        write_plot(args.plot, sweep, loo)
        print(f"\nPlot written to {args.plot}")


def write_plot(path: Path, sweep, loo) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if sweep:
        sizes = [n for n, _ in sweep]
        for key in MIN_FEATURE_TOLERANCES:
            series = [spread.get(key) for _, spread in sweep]
            if any(v is not None for v in series):
                axes[0].plot(sizes, series, marker="o", label=CATEGORY_LABEL[key])
        axes[0].set_xlabel("expert takes in envelope")
        axes[0].set_ylabel("mean standard deviation")
        axes[0].set_title("Envelope spread vs sample size")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

    if loo:
        values = [s for _, s in loo]
        axes[1].hist(values, bins=range(0, 105, 5), edgecolor="black")
        axes[1].set_xlabel("score of held-out expert take")
        axes[1].set_ylabel("takes")
        axes[1].set_title("Leave-one-out expert ceiling")
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
