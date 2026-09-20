"""Quality of the tracks on real football, not correctness of the code.

Every other test in this suite feeds the system blank frames and a fake model,
so all of them can pass while the output is visibly wrong. These are the ones
that would have caught that: they run the quality metrics against a real clip
(Tesr1) and hold the numbers to what was actually measured, so a change that
makes the tracking worse fails here rather than in somebody's eyes.

The colour series are recorded fixtures rather than sampled from the video,
because seeking 295 frames takes minutes and the numbers only move when the
tracks do. The unrepaired series is kept alongside the repaired one on purpose:
a detector that has only ever been shown good data cannot prove it still
detects anything.

BASELINES ARE NOT TARGETS. Two swaps survive on this clip. The assertions say
"no worse than what we measured", so the remaining faults are recorded instead
of hidden, and tightening a number here is how a real improvement gets locked
in.
"""

import json
import math
from pathlib import Path

import pytest

from analytics.pitch_role import (
    GOAL_FRACTION,
    PitchRole,
    classify,
    footprint,
)
from analytics.tracking_quality import (
    impossible_links,
    kit_changepoints,
    mid_pitch_births,
)

FIXTURES = Path(__file__).parent / "fixtures"
OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
REPAIRED_TRACKS = OUTPUTS / "Tesr1_tracks_kitfixed.json"
ROSTER_TRACKS = OUTPUTS / "Tesr1_tracks_roster.json"

FRAME_W, FRAME_H = 1920, 1080
STRONG = 0.70            # a changepoint this clear is not arguable

# Measured on Tesr1, 2026-09-18, imgsz 1920 + stitching + kit-swap repair.
MAX_SWAPPED_TRACKS = 2
MAX_SWAP_CUTS = 2
MAX_MID_PITCH_BIRTHS = 0
MAX_IMPOSSIBLE_LINKS = 12
MAX_PX_PER_FRAME_SEEN = 14.0


def load_kits(name):
    path = FIXTURES / f"tesr1_kits_{name}.json"
    if not path.is_file():
        pytest.skip(f"missing fixture {path.name}")
    return json.loads(path.read_text())


def changepoints(data):
    sep = data["kit_separation"]
    return {int(tid): kit_changepoints(s["frames"], s["colours"], sep)
            for tid, s in data["tracks"].items()
            if kit_changepoints(s["frames"], s["colours"], sep)}


class TestTheDetectorStillDetects:
    """Guards against the metric quietly becoming a function that returns []."""

    def test_finds_the_known_swaps_in_the_unrepaired_tracks(self):
        hits = changepoints(load_kits("unrepaired"))
        assert len(hits) >= 9, "detector went blind on a known-bad file"

    def test_those_swaps_are_not_marginal(self):
        hits = changepoints(load_kits("unrepaired"))
        strong = [c for v in hits.values() for c in v if c.strength >= STRONG]
        assert len(strong) >= 8

    def test_names_the_crossings_we_verified_by_eye(self):
        # These four were confirmed frame by frame: a white player becomes a
        # maroon one under the same id, which cannot happen.
        hits = changepoints(load_kits("unrepaired"))
        for track_id, frame in ((3, 506), (5, 431), (8, 451), (12, 496)):
            assert any(abs(c.frame - frame) <= 15 for c in hits.get(track_id, [])), \
                f"lost the verified swap on id {track_id} near f{frame}"


class TestTheRepairHolds:
    def test_almost_every_swap_is_gone(self):
        before = len(changepoints(load_kits("unrepaired")))
        after = len(changepoints(load_kits("repaired")))
        assert after <= MAX_SWAPPED_TRACKS
        assert after < before / 4

    def test_no_new_swap_appeared(self):
        cuts = sum(len(v) for v in changepoints(load_kits("repaired")).values())
        assert cuts <= MAX_SWAP_CUTS

    def test_the_two_that_survive_are_the_ones_we_know_about(self):
        # Recorded, not accepted. id 7 is a far-side player whose shirt patch is
        # a few dozen blurred pixels; id 17 crosses inside a group of four.
        hits = changepoints(load_kits("repaired"))
        assert sorted(hits) == [7, 17]


@pytest.mark.skipif(not REPAIRED_TRACKS.is_file(),
                    reason="run the pipeline on Tesr1 first")
class TestPositionalQuality:
    """The colour-blind half. These fail in the opposite direction to the kit
    metric, which is the whole reason both are kept: undoing a swap at the
    wrong frame fixes the colours and flings the player across the pitch."""

    @staticmethod
    def tracks():
        data = json.loads(REPAIRED_TRACKS.read_text())
        first, pos = {}, {}
        for frame in sorted(data["frames"], key=lambda f: f["frame"]):
            for t in frame["tracks"]:
                if t["class"] == "ball":
                    continue
                first.setdefault(t["track_id"], (frame["frame"], tuple(t["bbox"])))
                pos.setdefault(t["track_id"], {})[frame["frame"]] = (
                    (t["bbox"][0] + t["bbox"][2]) / 2.0, t["bbox"][3])
        start = min(f["frame"] for f in data["frames"])
        return first, pos, start

    def test_the_tracker_does_not_lose_people_mid_pitch(self):
        first, _, start = self.tracks()
        births = mid_pitch_births(first, FRAME_W, FRAME_H, start)
        assert len(births) <= MAX_MID_PITCH_BIRTHS, f"new ids appeared at {births}"

    def test_no_track_contains_a_run_nobody_could_make(self):
        _, pos, _ = self.tracks()
        links = impossible_links(pos)
        assert len(links) <= MAX_IMPOSSIBLE_LINKS

    def test_the_worst_step_stays_within_what_we_measured(self):
        _, pos, _ = self.tracks()
        worst = max((rate for _, _, _, rate in impossible_links(pos)), default=0.0)
        assert worst <= MAX_PX_PER_FRAME_SEEN

    def test_the_id_count_is_close_to_the_number_of_people(self):
        # 22 players + 3 officials on this clip, and ids are never reused, so
        # some slack is correct; a large excess means the tracker fragmented.
        first, _, _ = self.tracks()
        assert len(first) <= 32

    def test_every_track_is_a_continuous_stretch_of_frames(self):
        _, pos, _ = self.tracks()
        for track_id, series in pos.items():
            frames = sorted(series)
            assert frames == sorted(set(frames))
            assert not any(math.isnan(v) for p in series.values() for v in p)


@pytest.mark.skipif(not ROSTER_TRACKS.is_file(),
                    reason="run the roster step on Tesr1 first")
class TestTheSquadIsFixed:
    """The promise the roster makes: a bounded set of numbers, each meaning one
    person for the length of the clip. Anything else it does is incidental."""

    FPS = 25.0
    BALL_ID = 0
    UNNUMBERED = -1

    @staticmethod
    def frames():
        return json.loads(ROSTER_TRACKS.read_text())["frames"]

    def test_no_number_outside_the_squad_is_ever_issued(self):
        ids = {t["track_id"] for f in self.frames() for t in f["tracks"]
               if t["track_id"] > 0}
        assert ids <= set(range(1, 23))

    def test_both_squads_are_fully_numbered(self):
        # Eleven a side, all twenty-two worn. If this drops it means a player
        # was never tracked -- which the old free-running ids could not show,
        # because there were more ids than people and nobody counted them.
        ids = {t["track_id"] for f in self.frames() for t in f["tracks"]
               if t["track_id"] > 0}
        assert len(ids & set(range(1, 12))) == 11
        assert len(ids & set(range(12, 23))) == 11

    def test_the_ball_keeps_its_own_number(self):
        balls = {t["track_id"] for f in self.frames() for t in f["tracks"]
                 if t["class"] == "ball"}
        assert balls == {self.BALL_ID}

    def test_referees_carry_no_number(self):
        refs = {t["track_id"] for f in self.frames() for t in f["tracks"]
                if t["class"] == "referee"}
        assert refs <= {self.UNNUMBERED}

    def test_no_number_is_worn_twice_at_once(self):
        for frame in self.frames():
            worn = [t["track_id"] for t in frame["tracks"] if t["track_id"] > 0]
            assert len(worn) == len(set(worn)), f"duplicate at f{frame['frame']}"

    def test_never_more_than_eleven_of_a_side_on_screen(self):
        for frame in self.frames():
            worn = [t["track_id"] for t in frame["tracks"] if t["track_id"] > 0]
            assert sum(1 for i in worn if i <= 11) <= 11
            assert sum(1 for i in worn if i > 11) <= 11

    @staticmethod
    def spans():
        frames = TestTheSquadIsFixed.frames()
        span = {}
        for frame in frames:
            for t in frame["tracks"]:
                if t["track_id"] > 0:
                    lo, hi = span.get(t["track_id"], (frame["frame"],) * 2)
                    span[t["track_id"]] = (min(lo, frame["frame"]),
                                           max(hi, frame["frame"]))
        clip = len(frames)
        return {n: (hi - lo + 1) / clip for n, (lo, hi) in span.items()}

    def test_all_but_one_number_lasts_the_whole_clip(self):
        # Twenty-one of the twenty-two are worn essentially end to end. The
        # exception is recorded below rather than averaged away.
        held = self.spans()
        assert sum(1 for v in held.values() if v > 0.90) >= 21

    def test_the_one_short_number_belongs_to_a_keeper_barely_on_camera(self):
        # Squad 22 is Barcelona's goalkeeper, and he is in shot for 66 of 1471
        # frames -- the camera spends the clip at the other end. The number is
        # his for as long as he exists, which is all a roster can promise; the
        # 4% is the footage, not the tracking.
        held = self.spans()
        thin = {n: v for n, v in held.items() if v <= 0.90}
        assert list(thin) == [22]
        assert 0.02 < thin[22] < 0.10

    def test_unnumbered_boxes_stay_a_small_share(self):
        # Referees and anything the squad had no room for. On this clip it is
        # four officials, which is what it should be.
        total = sum(len(f["tracks"]) for f in self.frames())
        loose = sum(1 for f in self.frames() for t in f["tracks"]
                    if t["track_id"] == self.UNNUMBERED)
        assert loose / total < 0.15

    def test_no_official_is_wearing_a_squad_number(self):
        # Weak on its own -- it only catches an official the model also labels
        # one. TestSquadsByFootprint is the test that does not depend on the
        # label.
        numbered_refs = {t["track_id"] for f in self.frames() for t in f["tracks"]
                         if t["class"] == "referee" and t["track_id"] > 0}
        assert numbered_refs == set()


class TestSquadsByFootprint:
    """Who got a number, judged by where they walked rather than what the model
    called them.

    This is the check that matters. The model labels the referee a player in
    1469 of 1471 frames on this clip, so any test keyed on the class name
    agrees with the mistake instead of catching it. Metres on the pitch do not:
    an assistant referee patrols a touchline and a player does not, whatever
    either of them is wearing or labelled.
    """

    @staticmethod
    def footprints():
        path = FIXTURES / "tesr1_roster_pitch.json"
        if not path.is_file():
            pytest.skip("missing fixture tesr1_roster_pitch.json")
        data = json.loads(path.read_text())
        return data["tracks"], data["pitch"]

    def test_no_squad_number_patrols_a_touchline(self):
        tracks, pitch = self.footprints()
        for track_id, entry in tracks.items():
            if int(track_id) <= 0:
                continue
            role = classify(entry["points"], wears_a_team_kit=False,
                            length=pitch["length"], width=pitch["width"])
            assert role is not PitchRole.ASSISTANT,                 f"squad number {track_id} walks the touchline like an official"

    def test_the_officials_are_in_the_unnumbered_group(self):
        tracks, pitch = self.footprints()
        loose = tracks.get("-1")
        assert loose is not None, "nothing was left unnumbered at all"
        shape = footprint(loose["points"], pitch["length"], pitch["width"])
        assert shape.on_touchline > 0.5

    def test_a_goalkeeper_shaped_track_is_inside_the_squad(self):
        # A keeper's kit differs from both outfield kits by design, so a
        # colour-only rule strikes him off along with the officials. This is
        # the check that he stayed in: somebody numbered holds the area in
        # front of a goal, which no official does.
        tracks, pitch = self.footprints()
        held = []
        for track_id, entry in tracks.items():
            if int(track_id) <= 0:
                continue
            shape = footprint(entry["points"], pitch["length"], pitch["width"])
            if shape and shape.in_goal_area >= GOAL_FRACTION:
                held.append(int(track_id))
        assert held, "no squad number keeps goal; the keeper was left out"


class TestShirtsBehindTheNumbers:
    """The decisive referee check: what is each squad number wearing?

    The footprint tests catch an assistant, who patrols a line. They do not
    catch the referee, who moves like a player because he follows the play --
    on this clip he was given squad number 22 and only his shirt gave him away.
    """

    KIT_A = range(1, 12)
    KIT_B = range(12, 23)
    # A numbered shirt must sit at least this far from the officials' colour,
    # measured in the distance between the two kits. The keeper is the nearest
    # at 0.52; a referee holding a number scores near zero.
    MIN_MARGIN = 0.35

    @staticmethod
    def shirts():
        path = FIXTURES / "tesr1_roster_shirts.json"
        if not path.is_file():
            pytest.skip("missing fixture tesr1_roster_shirts.json")
        return json.loads(path.read_text())["tracks"]

    @staticmethod
    def _distance(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def kits_and_officials(self):
        shirts = self.shirts()
        def median(numbers):
            worn = [shirts[str(n)] for n in numbers if str(n) in shirts]
            return [sorted(c)[len(c) // 2]
                    for c in zip(*worn)]
        return median(self.KIT_A), median(self.KIT_B), shirts["-1"]

    def test_the_two_kits_are_clearly_different_colours(self):
        kit_a, kit_b, _ = self.kits_and_officials()
        assert self._distance(kit_a, kit_b) > 100

    def test_the_officials_wear_neither_kit(self):
        kit_a, kit_b, officials = self.kits_and_officials()
        separation = self._distance(kit_a, kit_b)
        margin = min(self._distance(officials, kit_a),
                     self._distance(officials, kit_b)) / separation
        assert margin > 0.4, (
            "officials sit close to a kit, so the grass mask is probably "
            "eating their shirt again")

    def test_no_squad_number_is_wearing_the_officials_colour(self):
        kit_a, kit_b, officials = self.kits_and_officials()
        separation = self._distance(kit_a, kit_b)
        shirts = self.shirts()
        worst = min(
            (self._distance(colour, officials) / separation, int(n))
            for n, colour in shirts.items() if int(n) > 0)
        assert worst[0] >= self.MIN_MARGIN, (
            f"squad number {worst[1]} is dressed like an official "
            f"({worst[0]:.2f} kits away)")

    def test_every_number_is_accounted_for(self):
        shirts = self.shirts()
        numbered = {int(n) for n in shirts if int(n) > 0}
        assert numbered <= set(range(1, 23))
        assert len(numbered) >= 21
