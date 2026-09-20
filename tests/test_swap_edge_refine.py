"""Tests for placing a swap's exchange at the crossing.

Appearance tells us *that* two tracks were exchanged; it cannot tell us *when*,
because a jersey only reads as the other team once the players have drawn apart
again. These cover the correction that moves the window's edges onto the
crossing, and the two cases where an edge must be left alone.
"""

import math

import pytest

from tracking.swap_corrector import join_cost, refine_swap_edges


def line(track_id, start, end, x0, y0, dx, dy, step=1):
    """Frames of a track moving at a constant rate."""
    return {f: (x0 + dx * (f - start), y0 + dy * (f - start))
            for f in range(start, end + 1, step)}


class TestJoinCost:
    def test_costs_only_the_travel_when_the_two_coincide(self):
        # Overlapping paths: the join costs the one frame of travel it spans,
        # nothing for the handover itself.
        a = line(1, 0, 20, 0.0, 0.0, 1.0, 0.0)
        b = line(2, 0, 20, 0.0, 0.0, 1.0, 0.0)
        assert join_cost(a, b, 10) == pytest.approx(1.0)

    def test_charges_the_distance_between_them(self):
        a = line(1, 0, 20, 0.0, 0.0, 1.0, 0.0)
        b = line(2, 0, 20, 0.0, 30.0, 1.0, 0.0)
        assert join_cost(a, b, 10) == pytest.approx(
            math.dist((9.0, 0.0), (10.0, 30.0)))

    def test_a_gap_in_the_track_is_spread_over_its_frames(self):
        # b is missing frames 8..12, so the join from a spans three frames and
        # the same displacement counts for a third as much per frame.
        a = {f: (float(f), 0.0) for f in range(0, 21)}
        b = {f: (float(f), 30.0) for f in list(range(0, 8)) + list(range(13, 21))}
        assert join_cost(a, b, 11) == pytest.approx(
            math.dist((10.0, 0.0), (13.0, 30.0)) / 3)

    def test_none_when_one_side_has_nothing_to_join(self):
        a = line(1, 0, 20, 0.0, 0.0, 1.0, 0.0)
        b = line(2, 0, 20, 0.0, 0.0, 1.0, 0.0)
        assert join_cost(a, b, 0) is None      # nothing before the frame
        assert join_cost(a, b, 21) is None     # nothing after it


class TestRefineSwapEdges:
    def test_moves_the_start_to_where_the_paths_meet(self):
        # The two cross at frame 50; appearance only noticed by frame 90.
        a = {f: (float(f), 100.0 - f) for f in range(0, 101)}
        b = {f: (float(f), f) for f in range(0, 101)}
        (_, _, start, end), = refine_swap_edges(
            [(1, 2, 90, 100)], {1: a, 2: b}, tol=45)
        assert start == 50
        assert end == 100

    def test_leaves_an_edge_that_is_the_start_of_both_tracks(self):
        # A window open from the first frame was never switched on mid-clip;
        # moving it would carve a swap out of a correct stretch.
        a = {f: (float(f), 100.0 - f) for f in range(0, 101)}
        b = {f: (float(f), f) for f in range(0, 101)}
        (_, _, start, _), = refine_swap_edges(
            [(1, 2, 0, 100)], {1: a, 2: b}, tol=45)
        assert start == 0

    def test_leaves_an_edge_that_is_the_end_of_both_tracks(self):
        a = {f: (float(f), 100.0 - f) for f in range(0, 101)}
        b = {f: (float(f), f) for f in range(0, 101)}
        (_, _, _, end), = refine_swap_edges(
            [(1, 2, 90, 100)], {1: a, 2: b}, tol=45)
        assert end == 100

    def test_moves_a_closing_edge_onto_the_second_crossing(self):
        # A temporary swap: well apart, cross at 30, apart again, cross back
        # at 70, apart. Both edges should leave the appearance boundary.
        def zig(sign):
            out = {}
            for f in range(0, 121):
                if f <= 20:
                    y = 40.0
                elif f <= 40:
                    y = 40.0 - 4.0 * (f - 20)      # crosses zero at 30
                elif f <= 60:
                    y = -40.0
                elif f <= 80:
                    y = -40.0 + 4.0 * (f - 60)     # crosses back at 70
                else:
                    y = 40.0
                out[f] = (float(f), sign * y)
            return out
        a, b = zig(1.0), zig(-1.0)
        (_, _, start, end), = refine_swap_edges(
            [(1, 2, 25, 75)], {1: a, 2: b}, tol=15)
        assert start == 30
        assert end == 69          # exchange runs up to, not past, the crossing

    def test_never_lets_the_window_invert(self):
        a = {f: (float(f), 0.0) for f in range(0, 101)}
        b = {f: (float(f), 0.0) for f in range(0, 101)}
        (_, _, start, end), = refine_swap_edges(
            [(1, 2, 50, 52)], {1: a, 2: b}, tol=45)
        assert start <= end

    def test_passes_through_a_pair_with_no_positions(self):
        assert refine_swap_edges([(1, 2, 10, 20)], {}, tol=45) == [(1, 2, 10, 20)]

    def test_keeps_the_ids_and_the_count(self):
        a = {f: (float(f), 100.0 - f) for f in range(0, 101)}
        b = {f: (float(f), f) for f in range(0, 101)}
        out = refine_swap_edges([(1, 2, 90, 100)], {1: a, 2: b}, tol=45)
        assert len(out) == 1
        assert out[0][0] == 1 and out[0][1] == 2
