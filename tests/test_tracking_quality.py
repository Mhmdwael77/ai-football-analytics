"""Tests for the tracking-quality measurements themselves.

Synthetic, like the rest of the suite -- these check that the metrics report
what they claim to. Whether the metrics are *good enough on real football* is a
separate question, answered by ``test_tracking_quality_real.py``, which runs
them against a real clip and recorded baselines.
"""

import pytest

from analytics.tracking_quality import (
    Changepoint,
    foot,
    impossible_links,
    kit_changepoints,
    mid_pitch_births,
)

W, H = 1920, 1080


def box_at(cx, cy, w=40.0, h=90.0):
    return (cx - w / 2, cy - h, cx + w / 2, cy)


class TestMidPitchBirths:
    def test_an_id_appearing_in_open_play_is_a_loss(self):
        births = {7: (400, box_at(960.0, 540.0))}
        assert mid_pitch_births(births, W, H, start_frame=1) == [7]

    def test_an_id_walking_in_from_the_touchline_is_not(self):
        births = {7: (400, box_at(40.0, 540.0))}
        assert mid_pitch_births(births, W, H, start_frame=1) == []

    def test_the_far_touchline_counts_as_an_edge_too(self):
        births = {7: (400, box_at(1900.0, 540.0))}
        assert mid_pitch_births(births, W, H, start_frame=1) == []

    def test_everyone_present_at_kickoff_is_not_a_birth(self):
        births = {1: (1, box_at(960.0, 540.0)), 2: (1, box_at(500.0, 400.0))}
        assert mid_pitch_births(births, W, H, start_frame=1) == []

    def test_reports_every_offender_in_id_order(self):
        births = {9: (300, box_at(900.0, 500.0)),
                  4: (500, box_at(700.0, 600.0)),
                  2: (600, box_at(10.0, 600.0))}
        assert mid_pitch_births(births, W, H, start_frame=1) == [4, 9]


class TestImpossibleLinks:
    def test_a_player_running_normally_is_clean(self):
        pos = {1: {f: (float(f) * 8.0, 500.0) for f in range(0, 50)}}
        assert impossible_links(pos) == []

    def test_a_teleport_is_caught(self):
        pos = {1: {10: (100.0, 500.0), 11: (900.0, 500.0)}}
        links = impossible_links(pos)
        assert [(t, a, b) for t, a, b, _ in links] == [(1, 10, 11)]

    def test_a_gap_is_judged_by_speed_not_by_distance(self):
        # 300 px across 60 frames is 5 px/frame -- a jog, not a teleport.
        pos = {1: {10: (100.0, 500.0), 70: (400.0, 500.0)}}
        assert impossible_links(pos) == []

    def test_the_same_distance_in_one_frame_is_a_teleport(self):
        pos = {1: {10: (100.0, 500.0), 11: (400.0, 500.0)}}
        assert len(impossible_links(pos)) == 1

    def test_reports_the_rate_so_severity_is_comparable(self):
        pos = {1: {10: (0.0, 0.0), 12: (0.0, 60.0)}}
        (_, _, _, rate), = impossible_links(pos)
        assert rate == pytest.approx(30.0)

    def test_the_threshold_is_adjustable_for_another_camera(self):
        pos = {1: {10: (0.0, 0.0), 11: (0.0, 20.0)}}
        assert impossible_links(pos, max_px_per_frame=11.0)
        assert impossible_links(pos, max_px_per_frame=25.0) == []


WHITE = [190.0, 215.0, 205.0]
MAROON = [80.0, 95.0, 115.0]
SEP = 186.0


def series(pattern, jitter=0.0):
    """A colour series from a string of W/M, one sample per character."""
    frames, cols = [], []
    for i, ch in enumerate(pattern):
        base = WHITE if ch == "W" else MAROON
        wob = jitter * (1 if i % 2 else -1)
        frames.append(i * 5)
        cols.append([c + wob for c in base])
    return frames, cols


class TestKitChangepoints:
    def test_one_kit_throughout_is_not_a_change(self):
        assert kit_changepoints(*series("W" * 40), SEP) == []

    def test_a_permanent_swap_is_found_where_it_happens(self):
        frames, cols = series("W" * 20 + "M" * 20)
        (cut,) = kit_changepoints(frames, cols, SEP)
        assert cut.frame == frames[20]
        assert cut.strength > 0.5

    def test_a_track_swapped_twice_gives_two_cuts(self):
        frames, cols = series("W" * 15 + "M" * 15 + "W" * 15)
        cuts = kit_changepoints(frames, cols, SEP)
        assert [c.frame for c in cuts] == [frames[15], frames[30]]

    def test_flicker_is_not_a_change(self):
        # This is what a far-side box does: alternating labels, no structure.
        # Run-counting would report a dozen swaps here.
        assert kit_changepoints(*series("WM" * 20), SEP) == []

    def test_a_brief_wobble_near_one_end_is_ignored(self):
        # Fewer samples than min_segment on one side: not enough to judge.
        assert kit_changepoints(*series("W" * 37 + "M" * 3), SEP) == []

    def test_noise_within_one_kit_is_not_a_change(self):
        frames, cols = series("W" * 40, jitter=25.0)
        assert kit_changepoints(frames, cols, SEP) == []

    def test_strength_rises_as_the_two_kits_separate(self):
        frames, cols = series("W" * 20 + "M" * 20)
        far, = kit_changepoints(frames, cols, SEP)
        near, = kit_changepoints(frames, cols, SEP * 1.6)
        assert far.strength > near.strength

    def test_too_short_a_series_returns_nothing(self):
        assert kit_changepoints(*series("W" * 5 + "M" * 5), SEP) == []

    def test_a_degenerate_separation_is_refused(self):
        frames, cols = series("W" * 20 + "M" * 20)
        assert kit_changepoints(frames, cols, 0.0) == []


class TestFoot:
    def test_is_the_bottom_centre_of_the_box(self):
        assert foot((100.0, 200.0, 140.0, 290.0)) == (120.0, 290.0)
