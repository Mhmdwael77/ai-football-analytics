"""Tests for placing tracks into a fixed squad of numbered shirts."""

import pytest

from tracking.models import FrameTracks, Track
from tracking.roster import (
    BALL_ID,
    UNASSIGNED_ID,
    TrackSpan,
    apply,
    apply_roster,
    assign_slots,
    build_spans,
    overflow_frames,
)


def span(track_id, team, first, last, x0=100.0, x1=None, y=500.0):
    return TrackSpan(track_id=track_id, team=team, first_frame=first,
                     last_frame=last, first_point=(x0, y),
                     last_point=(x0 if x1 is None else x1, y))


def track(frame, track_id, name="player", x=100.0, y=500.0):
    return Track(frame_id=frame, track_id=track_id, class_id=0, class_name=name,
                 confidence=0.9, bbox=(x - 20, y - 90, x + 20, y))


class TestSquadNumbering:
    def test_the_first_team_takes_one_to_eleven(self):
        spans = [span(i, 0, 1, 100, x0=100.0 * i) for i in range(1, 12)]
        assignment, overflow = assign_slots(spans)
        assert overflow == []
        assert sorted(assignment.values()) == list(range(1, 12))

    def test_the_second_team_takes_twelve_to_twenty_two(self):
        spans = [span(i, 1, 1, 100, x0=100.0 * i) for i in range(1, 12)]
        assignment, _ = assign_slots(spans)
        assert sorted(assignment.values()) == list(range(12, 23))

    def test_a_number_is_never_shared_across_teams(self):
        spans = ([span(i, 0, 1, 100, x0=100.0 * i) for i in range(1, 12)]
                 + [span(100 + i, 1, 1, 100, x0=100.0 * i) for i in range(1, 12)])
        assignment, _ = assign_slots(spans)
        first = {v for k, v in assignment.items() if k < 100}
        second = {v for k, v in assignment.items() if k >= 100}
        assert not (first & second)
        assert len(first) == len(second) == 11


class TestOnePersonPerSlot:
    def test_tracks_alive_together_get_different_numbers(self):
        spans = [span(1, 0, 1, 500), span(2, 0, 100, 600)]
        assignment, _ = assign_slots(spans)
        assert assignment[1] != assignment[2]

    def test_a_track_that_ends_frees_its_number(self):
        spans = [span(1, 0, 1, 100), span(2, 0, 120, 300)]
        assignment, _ = assign_slots(spans)
        assert assignment[1] == assignment[2]

    def test_touching_spans_do_not_share(self):
        # One ends on the frame the next begins: they were both there.
        spans = [span(1, 0, 1, 100), span(2, 0, 100, 300)]
        assignment, _ = assign_slots(spans)
        assert assignment[1] != assignment[2]


class TestContinuity:
    def test_a_slot_prefers_the_player_it_just_had(self):
        # Track 1 ends at x=800; two slots are free, and the returning player
        # reappears next to where track 1 left off.
        spans = [span(1, 0, 1, 100, x0=800.0),
                 span(2, 0, 1, 100, x0=100.0),
                 span(3, 0, 130, 400, x0=810.0)]
        assignment, _ = assign_slots(spans)
        assert assignment[3] == assignment[1]

    def test_a_distant_reappearance_does_not_steal_the_number(self):
        spans = [span(1, 0, 1, 100, x0=800.0),
                 span(2, 0, 130, 400, x0=100.0)]
        assignment, _ = assign_slots(spans, distance_gate_px=250.0)
        assert assignment[2] != assignment[1]

    def test_a_long_absence_does_not_keep_the_number(self):
        spans = [span(1, 0, 1, 100, x0=800.0),
                 span(2, 0, 400, 600, x0=805.0)]
        assignment, _ = assign_slots(spans, max_gap_frames=75)
        assert assignment[2] != assignment[1]

    def test_an_absence_within_the_window_keeps_it(self):
        spans = [span(1, 0, 1, 100, x0=800.0),
                 span(2, 0, 160, 400, x0=805.0)]
        assignment, _ = assign_slots(spans, max_gap_frames=75)
        assert assignment[2] == assignment[1]


class TestOverflow:
    def test_a_twelfth_simultaneous_player_gets_no_number(self):
        spans = [span(i, 0, 1, 500, x0=100.0 * i) for i in range(1, 13)]
        assignment, overflow = assign_slots(spans)
        assert len(assignment) == 11
        assert overflow == [12]

    def test_overflow_does_not_corrupt_the_others(self):
        spans = [span(i, 0, 1, 500, x0=100.0 * i) for i in range(1, 13)]
        assignment, _ = assign_slots(spans)
        assert sorted(assignment.values()) == list(range(1, 12))

    def test_the_squad_recovers_once_somebody_leaves(self):
        spans = ([span(i, 0, 1, 200, x0=100.0 * i) for i in range(1, 12)]
                 + [span(99, 0, 250, 400, x0=9000.0)])
        assignment, overflow = assign_slots(spans)
        assert overflow == []
        assert 99 in assignment

    def test_names_the_frames_where_the_squad_ran_short(self):
        frames = [FrameTracks(f, [track(f, 12)]) for f in (10, 11, 12)]
        assert overflow_frames(frames, [12]) == [10, 11, 12]


class TestBuildSpans:
    def test_reads_first_and_last_sighting(self):
        frames = [FrameTracks(f, [track(f, 7, x=10.0 * f)]) for f in (5, 9, 30)]
        (s,) = build_spans(frames, {7: 0})
        assert (s.first_frame, s.last_frame) == (5, 30)
        assert s.first_point[0] == pytest.approx(50.0)
        assert s.last_point[0] == pytest.approx(300.0)

    def test_skips_the_ball(self):
        frames = [FrameTracks(1, [track(1, 7), track(1, 8, name="ball")])]
        assert [s.track_id for s in build_spans(frames, {7: 0, 8: 0})] == [7]

    def test_skips_a_track_with_no_known_team(self):
        frames = [FrameTracks(1, [track(1, 7), track(1, 8)])]
        assert [s.track_id for s in build_spans(frames, {7: 0})] == [7]


class TestApplyRoster:
    def test_rewrites_player_ids_to_squad_numbers(self):
        frames = [FrameTracks(1, [track(1, 501)])]
        out = apply_roster(frames, {501: 4})
        assert out[0].tracks[0].track_id == 4

    def test_the_ball_always_gets_its_own_number(self):
        frames = [FrameTracks(1, [track(1, 900, name="ball")])]
        out = apply_roster(frames, {})
        assert out[0].tracks[0].track_id == BALL_ID

    def test_referees_are_left_unnumbered(self):
        frames = [FrameTracks(1, [track(1, 40, name="referee")])]
        out = apply_roster(frames, {40: 3})
        assert out[0].tracks[0].track_id == UNASSIGNED_ID

    def test_a_track_with_no_slot_is_left_unnumbered(self):
        frames = [FrameTracks(1, [track(1, 77)])]
        out = apply_roster(frames, {})
        assert out[0].tracks[0].track_id == UNASSIGNED_ID

    def test_boxes_and_classes_survive_the_rewrite(self):
        frames = [FrameTracks(1, [track(1, 501, x=333.0)])]
        out = apply_roster(frames, {501: 4})
        assert out[0].tracks[0].bbox == (313.0, 410.0, 353.0, 500.0)
        assert out[0].tracks[0].class_name == "player"


class TestApply:
    def test_end_to_end_gives_a_bounded_set_of_ids(self):
        frames = []
        for f in range(1, 200):
            tracks = [track(f, i, x=60.0 * i) for i in range(1, 12)]
            tracks += [track(f, 100 + i, x=60.0 * i, y=700.0) for i in range(1, 12)]
            tracks.append(track(f, 900, name="ball", x=500.0))
            tracks.append(track(f, 800, name="referee", x=900.0))
            frames.append(FrameTracks(f, tracks))
        team_of = {i: 0 for i in range(1, 12)}
        team_of.update({100 + i: 1 for i in range(1, 12)})

        result = apply(frames, team_of)
        ids = {t.track_id for fr in result.frames for t in fr.tracks}
        assert ids == set(range(1, 23)) | {BALL_ID, UNASSIGNED_ID}
        assert result.overflow == []
        assert result.slots_used == {0: 11, 1: 11}

    def test_every_frame_survives(self):
        frames = [FrameTracks(f, [track(f, 1)]) for f in range(1, 50)]
        result = apply(frames, {1: 0})
        assert [f.frame_index for f in result.frames] == list(range(1, 50))
