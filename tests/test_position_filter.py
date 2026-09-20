"""Tests for the physically-constrained position filter."""

from __future__ import annotations

import math

import pytest

from analytics.position_filter import (
    FilterStats,
    MAX_ACCEL_MS2,
    filter_track,
    filter_tracks,
)


def _velocities(points, fps=25.0):
    out = []
    for (f0, (x0, y0)), (f1, (x1, y1)) in zip(points, points[1:]):
        dt = (f1 - f0) / fps
        out.append(((x1 - x0) / dt, (y1 - y0) / dt))
    return out


def _speeds(points, fps=25.0):
    return [math.hypot(vx, vy) for vx, vy in _velocities(points, fps)]


def _accels(points, fps=25.0):
    """Acceleration from the velocity VECTOR.

    Using the speed magnitude instead would report zero for an oscillation
    that reverses direction every frame -- exactly the box jitter we care
    about, whose speed is constant and whose direction flips.
    """
    v = _velocities(points, fps)
    return [math.hypot(bx - ax, by - ay) * fps
            for (ax, ay), (bx, by) in zip(v, v[1:])]


# ---------------------------------------------------------------------------
# it must not touch honest motion
# ---------------------------------------------------------------------------
def test_a_player_running_at_a_steady_speed_is_left_alone():
    """8 m/s in a straight line is a real sprint, not noise."""
    fps = 25.0
    samples = [(i, (i * 8.0 / fps, 34.0)) for i in range(20)]

    out = filter_track(samples, fps=fps)

    for (_, (x, y)), (_, (ex, ey)) in zip(out, samples):
        assert x == pytest.approx(ex, abs=1e-6)
        assert y == pytest.approx(ey, abs=1e-6)


def test_a_hard_but_possible_turn_survives():
    """Changing direction at ~10 m/s^2 is within a footballer's range."""
    fps = 25.0
    samples = []
    x, v = 0.0, 5.0
    for i in range(20):
        samples.append((i, (x, 34.0)))
        v = max(v - 10.0 / fps, -5.0)          # decelerate at 10 m/s^2
        x += v / fps

    out = filter_track(samples, fps=fps)
    accel = _accels(out, fps)

    assert max(accel) <= MAX_ACCEL_MS2 + 1e-6
    # and it really did turn around rather than being frozen
    xs = [p[1][0] for p in out]
    assert max(xs) - min(xs) > 0.5


# ---------------------------------------------------------------------------
# it must remove what physics forbids
# ---------------------------------------------------------------------------
def test_box_jitter_is_suppressed():
    """A stationary player whose box wobbles must not gain acceleration.

    This is the real-world case: the detection box breathes as the player
    moves their arms, and the raw positions oscillate a few centimetres every
    frame. Differentiated twice that is hundreds of m/s^2.
    """
    fps = 25.0
    samples = [(i, (52.5 + (0.12 if i % 2 else -0.12), 34.0)) for i in range(30)]

    raw_accel = _accels(samples, fps)
    out = filter_track(samples, fps=fps)
    accel = _accels(out, fps)

    assert max(raw_accel) > 100, "the raw signal really is impossible"
    assert max(accel) <= MAX_ACCEL_MS2 + 1e-6


def test_a_teleport_is_clamped_to_a_reachable_step():
    """An id switch puts the position on another player; it cannot be followed."""
    fps = 25.0
    samples = [(i, (10.0, 34.0)) for i in range(5)]
    samples.append((5, (60.0, 34.0)))          # 50 m in one frame

    out = filter_track(samples, fps=fps)
    step = math.dist(out[-1][1], out[-2][1])

    assert step < 0.6, "one frame at 25 fps cannot cover more than ~0.4 m"


def test_speeds_stay_within_human_limits():
    fps = 25.0
    samples = [(i, (i * 40.0 / fps, 34.0)) for i in range(15)]   # 40 m/s

    out = filter_track(samples, fps=fps)

    assert max(_speeds(out, fps)) <= 10.0 + 1e-6


# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------
def test_a_long_gap_starts_a_new_segment_instead_of_inventing_a_path():
    fps = 25.0
    samples = [(0, (10.0, 34.0)), (1, (10.4, 34.0)),
               (40, (70.0, 20.0)), (41, (70.3, 20.0))]

    stats = FilterStats()
    out = filter_track(samples, fps=fps, stats=stats)

    assert stats.segments == 2
    # the first point after the gap is taken as measured, not dragged
    assert out[2][1] == (70.0, 20.0)


def test_a_single_sample_is_returned_untouched():
    assert filter_track([(3, (1.0, 2.0))]) == [(3, (1.0, 2.0))]


def test_an_empty_track_is_fine():
    assert filter_track([]) == []


def test_samples_come_back_in_frame_order():
    out = filter_track([(5, (1.0, 1.0)), (1, (0.0, 0.0)), (3, (0.5, 0.5))])
    assert [f for f, _ in out] == [1, 3, 5]


def test_a_zero_or_negative_fps_is_rejected():
    with pytest.raises(ValueError):
        filter_track([(0, (0.0, 0.0)), (1, (1.0, 1.0))], fps=0)


# ---------------------------------------------------------------------------
# stats and the multi-track wrapper
# ---------------------------------------------------------------------------
def test_stats_report_how_much_had_to_be_clamped():
    fps = 25.0
    jitter = [(i, (52.5 + (0.12 if i % 2 else -0.12), 34.0)) for i in range(30)]

    stats = FilterStats()
    filter_track(jitter, fps=fps, stats=stats)

    assert stats.points == 30
    assert stats.accel_clamped > 20
    assert stats.clamped_fraction > 0.5


def test_clean_motion_clamps_nothing():
    fps = 25.0
    clean = [(i, (i * 4.0 / fps, 34.0)) for i in range(20)]

    stats = FilterStats()
    filter_track(clean, fps=fps, stats=stats)

    assert stats.accel_clamped == 0
    assert stats.speed_clamped == 0
    assert stats.clamped_fraction == 0.0


def test_filter_tracks_handles_every_track_and_shares_stats():
    fps = 25.0
    tracks = {
        1: [(i, (i * 3.0 / fps, 30.0)) for i in range(10)],
        2: [(i, (52.5 + (0.15 if i % 2 else -0.15), 34.0)) for i in range(10)],
    }

    out, stats = filter_tracks(tracks, fps=fps)

    assert set(out) == {1, 2}
    assert stats.points == 20
    assert stats.accel_clamped > 0, "track 2's jitter must have been clamped"
