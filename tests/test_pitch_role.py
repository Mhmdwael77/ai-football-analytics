"""Tests for reading a role off where a track walks.

The cases are drawn from the shapes measured on Tesr1: an assistant pinned to
y=69, a keeper holding x=91..104 across the middle of the goal, a referee
covering 439 m^2 of the centre.
"""

import pytest

from analytics.pitch_role import (
    GOAL_FRACTION,
    MIN_POINTS,
    best_deepest_share,
    deepest_share,
    PITCH_LENGTH,
    PITCH_WIDTH,
    Footprint,
    PitchRole,
    belongs_in_a_squad,
    classify,
    footprint,
)


def walk(x0, y0, x1, y1, n=30):
    """n points along a straight line, as a track that moved."""
    return [(x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1))
            for i in range(n)]


ASSISTANT = walk(74.0, 69.0, 93.0, 69.5)          # the far touchline
ASSISTANT_NEAR = walk(38.0, 0.5, 52.0, 1.5)       # the near one
KEEPER = walk(91.0, 26.0, 104.0, 34.0)            # in front of a goal
KEEPER_OTHER_END = walk(1.0, 30.0, 14.0, 38.0)
REFEREE = walk(52.0, 16.0, 76.0, 35.0)            # the middle


class TestFootprint:
    def test_too_few_points_gives_nothing(self):
        assert footprint(walk(50.0, 34.0, 55.0, 34.0, n=MIN_POINTS - 1)) is None

    def test_enough_points_gives_a_shape(self):
        assert footprint(walk(50.0, 34.0, 55.0, 34.0, n=MIN_POINTS)) is not None

    def test_drops_points_that_failed_to_project(self):
        pts = walk(50.0, 34.0, 55.0, 34.0, n=20) + [(float("nan"), 3.0)] * 5
        shape = footprint(pts)
        assert shape.n_points == 20

    def test_measures_the_extent_it_covered(self):
        shape = footprint(walk(40.0, 10.0, 60.0, 30.0))
        assert shape.x_min == pytest.approx(40.0)
        assert shape.x_max == pytest.approx(60.0)
        assert shape.area == pytest.approx(400.0)

    def test_measures_the_distance_walked(self):
        shape = footprint(walk(0.0, 0.0, 30.0, 40.0))
        assert shape.travel == pytest.approx(50.0)

    def test_a_touchline_walk_reads_as_on_the_line(self):
        assert footprint(ASSISTANT).on_touchline == pytest.approx(1.0)

    def test_a_walk_through_the_middle_does_not(self):
        assert footprint(REFEREE).on_touchline == pytest.approx(0.0)

    def test_standing_in_front_of_a_goal_reads_as_such(self):
        assert footprint(KEEPER).in_goal_area > GOAL_FRACTION

    def test_the_far_goal_counts_the_same_as_the_near_one(self):
        assert footprint(KEEPER_OTHER_END).in_goal_area > GOAL_FRACTION


class TestClassify:
    def test_a_team_shirt_settles_it_without_geometry(self):
        # Even standing where a keeper stands: the kit says outfield.
        assert classify(KEEPER, wears_a_team_kit=True) is PitchRole.OUTFIELD

    def test_the_far_touchline_is_an_assistant(self):
        assert classify(ASSISTANT, wears_a_team_kit=False) is PitchRole.ASSISTANT

    def test_the_near_touchline_is_too(self):
        assert classify(ASSISTANT_NEAR, wears_a_team_kit=False) is PitchRole.ASSISTANT

    def test_in_front_of_a_goal_is_the_goalkeeper(self):
        assert classify(KEEPER, wears_a_team_kit=False) is PitchRole.GOALKEEPER

    def test_roaming_the_middle_is_the_referee(self):
        assert classify(REFEREE, wears_a_team_kit=False) is PitchRole.REFEREE

    def test_an_assistant_at_the_corner_is_not_mistaken_for_a_keeper(self):
        # Close to a goal line, but hugging the edge rather than the middle.
        corner = walk(96.0, 67.5, 104.0, 68.5)
        assert classify(corner, wears_a_team_kit=False) is PitchRole.ASSISTANT

    def test_too_short_a_track_gets_no_verdict(self):
        short = walk(91.0, 30.0, 95.0, 32.0, n=MIN_POINTS - 1)
        assert classify(short, wears_a_team_kit=False) is PitchRole.UNKNOWN

    def test_a_keeper_who_comes_out_for_a_corner_is_still_a_keeper(self):
        # Most of the track in the box, a short excursion upfield.
        pts = walk(92.0, 28.0, 102.0, 36.0, n=24) + walk(70.0, 34.0, 60.0, 34.0, n=6)
        assert classify(pts, wears_a_team_kit=False) is PitchRole.GOALKEEPER

    def test_the_touchline_moves_with_the_pitch_width(self):
        # y=65 is a metre off the line on a standard 68 m pitch and fifteen
        # metres inside one that is 80 m wide -- an assistant in the first case
        # and somebody in open play in the second, from identical coordinates.
        line = walk(40.0, 65.0, 60.0, 65.5)
        assert classify(line, wears_a_team_kit=False) is PitchRole.ASSISTANT
        assert classify(line, wears_a_team_kit=False,
                        width=80.0) is PitchRole.REFEREE

    def test_someone_standing_off_the_pitch_is_an_assistant(self):
        assert classify(walk(40.0, 70.0, 60.0, 71.0),
                        wears_a_team_kit=False) is PitchRole.ASSISTANT


class TestBelongsInASquad:
    def test_the_keeper_is_one_of_the_eleven(self):
        assert belongs_in_a_squad(PitchRole.GOALKEEPER)

    def test_so_is_an_outfield_player(self):
        assert belongs_in_a_squad(PitchRole.OUTFIELD)

    def test_officials_are_not(self):
        assert not belongs_in_a_squad(PitchRole.REFEREE)
        assert not belongs_in_a_squad(PitchRole.ASSISTANT)

    def test_an_unreadable_track_is_left_out_rather_than_guessed(self):
        # Costs a box on the overlay; the other way costs a player a number.
        assert not belongs_in_a_squad(PitchRole.UNKNOWN)

    def test_every_role_has_an_answer(self):
        for role in PitchRole:
            assert isinstance(belongs_in_a_squad(role), bool)


def line_frames(x0, x1, y, n=30, start=0, step=5):
    """A track as {frame: point}, walking along the pitch."""
    return {start + i * step: (x0 + (x1 - x0) * i / (n - 1), y)
            for i in range(n)}


def team_frames(xs, frames):
    """A team standing at fixed x positions, present on every given frame."""
    return {f: [(x, 34.0) for x in xs] for f in frames}


class TestDeepestShare:
    def test_a_track_behind_everyone_scores_one(self):
        track = line_frames(10.0, 12.0, 34.0)
        team = team_frames([30.0, 40.0, 50.0, 60.0, 70.0, 80.0], track)
        assert deepest_share(track, team, side=-1) == pytest.approx(1.0)

    def test_the_same_track_scores_zero_at_the_other_end(self):
        track = line_frames(10.0, 12.0, 34.0)
        team = team_frames([30.0, 40.0, 50.0, 60.0, 70.0, 80.0], track)
        assert deepest_share(track, team, side=+1) == pytest.approx(0.0)

    def test_a_track_in_among_them_scores_zero(self):
        track = line_frames(55.0, 57.0, 34.0)
        team = team_frames([30.0, 40.0, 50.0, 60.0, 70.0, 80.0], track)
        assert deepest_share(track, team, side=-1) == pytest.approx(0.0)

    def test_frames_with_too_few_of_the_team_are_not_counted(self):
        # Deeper than the two players currently in shot proves nothing.
        track = line_frames(10.0, 12.0, 34.0, n=10)
        team = {f: [(30.0, 34.0), (40.0, 34.0)] for f in track}
        assert deepest_share(track, team, side=-1) == pytest.approx(0.0)

    def test_a_partial_spell_in_front_lowers_the_score(self):
        track = line_frames(10.0, 10.0, 34.0, n=10)
        for i, f in enumerate(sorted(track)):
            if i >= 5:
                track[f] = (55.0, 34.0)      # steps up into the line
        team = team_frames([30.0, 40.0, 50.0, 60.0, 70.0, 80.0], track)
        assert deepest_share(track, team, side=-1) == pytest.approx(0.5)

    def test_no_shared_frames_gives_no_score(self):
        track = line_frames(10.0, 12.0, 34.0, start=0)
        team = team_frames([30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                           line_frames(0.0, 1.0, 0.0, start=9000))
        assert deepest_share(track, team, side=-1) == pytest.approx(0.0)

    def test_best_over_both_teams_and_both_ends(self):
        track = line_frames(10.0, 12.0, 34.0)
        near = team_frames([30.0, 40.0, 50.0, 60.0, 70.0, 80.0], track)
        far = team_frames([25.0, 35.0, 45.0, 55.0, 65.0, 75.0], track)
        assert best_deepest_share(track, [near, far]) == pytest.approx(1.0)

    def test_best_is_zero_when_nobody_is_behind_anybody(self):
        track = line_frames(55.0, 57.0, 34.0)
        team = team_frames([30.0, 40.0, 50.0, 60.0, 70.0, 80.0], track)
        assert best_deepest_share(track, [team]) == pytest.approx(0.0)


class TestTheRelativeKeeperTest:
    """The case that broke the absolute one: a keeper 27 m off his line, at an
    end of the pitch the camera never shows."""

    HIGH_LINE = [(27.0 + i * 0.3, 33.5 + (i % 3) * 0.5) for i in range(14)]

    def test_a_high_keeper_is_caught_by_being_behind_his_side(self):
        assert classify(self.HIGH_LINE, wears_a_team_kit=False,
                        behind_team=1.0) is PitchRole.GOALKEEPER

    def test_and_is_missed_without_it(self):
        # This is exactly what happened: no goal area, so only a referee left.
        assert classify(self.HIGH_LINE, wears_a_team_kit=False,
                        behind_team=0.0) is PitchRole.REFEREE

    def test_an_assistant_beyond_the_last_defender_is_still_an_assistant(self):
        # Measured at 100% deepest on Tesr1 -- the touchline test is what stops
        # the relative test handing him a squad number.
        assistant = walk(74.0, 69.0, 93.0, 69.5)
        assert classify(assistant, wears_a_team_kit=False,
                        behind_team=1.0) is PitchRole.ASSISTANT

    def test_a_wide_defender_who_is_deepest_is_not_a_keeper(self):
        # A full-back holding the last line down a flank: deep, but 17 m off
        # the midline, where a keeper sits within a few metres of it.
        flank = [(30.0 + i * 0.4, 17.0 + (i % 2)) for i in range(14)]
        assert classify(flank, wears_a_team_kit=False,
                        behind_team=1.0) is PitchRole.REFEREE

    def test_being_central_alone_is_not_enough(self):
        centre = [(50.0 + i * 0.5, 34.0) for i in range(14)]
        assert classify(centre, wears_a_team_kit=False,
                        behind_team=0.5) is PitchRole.REFEREE

    def test_the_goal_area_route_still_works_on_its_own(self):
        assert classify(KEEPER, wears_a_team_kit=False,
                        behind_team=0.0) is PitchRole.GOALKEEPER

    def test_a_keeper_found_this_way_belongs_in_a_squad(self):
        role = classify(self.HIGH_LINE, wears_a_team_kit=False, behind_team=1.0)
        assert belongs_in_a_squad(role)


class TestCentreOffset:
    def test_a_keeper_sits_on_the_midline(self):
        assert footprint(KEEPER).centre_offset < 6.0

    def test_an_assistant_is_as_far_from_it_as_you_can_get(self):
        assert footprint(ASSISTANT).centre_offset > 30.0

    def test_one_trip_upfield_does_not_move_the_median(self):
        pts = [(92.0, 34.0)] * 20 + [(60.0, 5.0)] * 4
        assert footprint(pts).centre_offset == pytest.approx(0.0)
