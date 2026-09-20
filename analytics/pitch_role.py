"""Tell a goalkeeper from a match official by where they stand, not what they wear.

Colour cannot do this and never will. A goalkeeper's kit is required to differ
from both outfield kits, which is exactly the test a "not either team's shirt"
rule applies to officials -- so any rule that filters officials by colour will
eventually throw the keeper out with them. On this clip the keeper wears pale
cyan, the referee yellow, and neither is white or maroon; nothing in the colour
tells you which of the two is a player.

Where they stand does. Over a passage of play the three jobs trace three
different shapes on the ground:

  an assistant referee runs *along a touchline*, so the whole track sits within
  a couple of metres of y=0 or y=68 and barely moves inwards;

  a goalkeeper stays *in front of one goal*: close to a goal line in x, and
  near the middle of the pitch in y, because that is where the goal is;

  the referee covers the middle and a great deal of it -- the largest
  footprint of anyone on the field.

These are metres on the pitch, so they mean the same thing at any camera angle
and in any match, which a pixel rule would not. They need a homography, and a
track whose points could not be projected simply gets no verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

Point = Tuple[float, float]

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# An assistant hugs the line; this is how far off it they may stray.
TOUCHLINE_BAND_M = 4.0
# How much of a track must sit in that band before we believe it.
TOUCHLINE_FRACTION = 0.80
# A keeper works within this of a goal line, and this of the pitch's midline.
GOAL_BOX_DEPTH_M = 22.0
GOAL_BOX_HALF_WIDTH_M = 16.0
GOAL_FRACTION = 0.70
# The relative test, for a keeper the absolute one cannot reach. How much of
# his time he spends behind every outfield player of one team...
DEEPEST_FRACTION = 0.80
# ...and how far his typical position strays from the middle of the pitch,
# which is what separates him from a full-back doing the same thing down a
# flank. Measured on Tesr1: keepers 0.7, 4.5 and 6.7 m off the midline; the
# widest outfielder who was ever deepest, 16.4 m; the assistants, 31 m and up.
KEEPER_CENTRE_BAND_M = 14.0
# How many of a team must be on screen before "deeper than all of them" means
# anything.
MIN_TEAM_VISIBLE = 6
# Below this many projected points there is not enough of a shape to judge.
MIN_POINTS = 8


class PitchRole(Enum):
    """What the footprint says this track does."""

    GOALKEEPER = "goalkeeper"
    ASSISTANT = "assistant"      # linesman: runs a touchline
    REFEREE = "referee"          # roams the middle
    OUTFIELD = "outfield"
    UNKNOWN = "unknown"          # too few points to say


@dataclass
class Footprint:
    """What a track did on the ground, in metres."""

    n_points: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    on_touchline: float          # fraction of points within the band
    in_goal_area: float          # fraction of points in front of a goal
    centre_offset: float         # typical distance from the pitch's midline, m
    area: float                  # bounding-box area, m^2
    travel: float                # path length, m


def footprint(points: Sequence[Point],
              length: float = PITCH_LENGTH,
              width: float = PITCH_WIDTH) -> Optional[Footprint]:
    """Summarise a track's pitch positions, or None if there are too few."""
    pts = [p for p in points
           if p[0] == p[0] and p[1] == p[1]]        # drop NaNs
    if len(pts) < MIN_POINTS:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mid = width / 2.0

    near_line = sum(
        1 for _, y in pts
        if y <= TOUCHLINE_BAND_M or y >= width - TOUCHLINE_BAND_M)
    near_goal = sum(
        1 for x, y in pts
        if (x <= GOAL_BOX_DEPTH_M or x >= length - GOAL_BOX_DEPTH_M)
        and abs(y - mid) <= GOAL_BOX_HALF_WIDTH_M)

    travel = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                 for a, b in zip(pts, pts[1:]))
    # Median, not mean: a keeper who comes up for one corner should not read as
    # a wide player for the rest of the match.
    offsets = sorted(abs(y - mid) for y in ys)
    return Footprint(
        n_points=len(pts),
        x_min=min(xs), x_max=max(xs), y_min=min(ys), y_max=max(ys),
        on_touchline=near_line / len(pts),
        in_goal_area=near_goal / len(pts),
        centre_offset=offsets[len(offsets) // 2],
        area=(max(xs) - min(xs)) * (max(ys) - min(ys)),
        travel=travel,
    )


def deepest_share(
    track: Mapping[int, Point],
    team: Mapping[int, Sequence[Point]],
    side: int,
    min_visible: int = MIN_TEAM_VISIBLE,
) -> float:
    """How much of this track's time is spent behind a whole team's outfield.

    ``side`` is -1 for the low-x end of the pitch and +1 for the high-x end.
    Frames where too few of the team are on screen are skipped rather than
    counted either way: being deeper than the three players currently in shot
    means nothing.

    This is the test that reaches a goalkeeper the goal-area test cannot. It
    does not care how high he plays, and it does not care where the camera is
    pointing, because it asks only where he is *relative to his own side* --
    and on this footage the near goal is never in frame at all, so nothing
    absolute could have worked.
    """
    hits = counted = 0
    for frame, point in track.items():
        mates = [p[0] for p in team.get(frame, ())]
        if len(mates) < min_visible:
            continue
        counted += 1
        if (point[0] < min(mates)) if side < 0 else (point[0] > max(mates)):
            hits += 1
    return hits / counted if counted else 0.0


def best_deepest_share(
    track: Mapping[int, Point],
    teams: Sequence[Mapping[int, Sequence[Point]]],
    min_visible: int = MIN_TEAM_VISIBLE,
) -> float:
    """The strongest "behind everyone" score over both teams and both ends.

    A keeper standing in front of his own goal is behind his own side, and
    usually behind the opposition too, so this does not identify whose keeper
    he is -- only that he is one. Which squad he joins is decided elsewhere,
    by his kit.
    """
    return max((deepest_share(track, team, side, min_visible)
                for team in teams for side in (-1, 1)), default=0.0)


def classify(
    points: Sequence[Point],
    wears_a_team_kit: bool,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
    behind_team: float = 0.0,
) -> PitchRole:
    """Read a role off the track's footprint.

    ``wears_a_team_kit`` is the colour evidence, and it is used only where it
    is trustworthy: someone dressed as one of the two teams is an outfield
    player and needs no geometry. Everyone else -- keeper, referee, assistant,
    all of whom wear something else on purpose -- is separated by the shape of
    their movement alone.

    ``behind_team`` is :func:`best_deepest_share`, and it is the second way in
    to being a goalkeeper. The goal-area test is absolute -- within so many
    metres of a goal line -- which fails on two counts at once for a keeper who
    plays a high line at an end the camera never shows: he is genuinely 27 m
    out, and there is no footage of the 22 m the test looks at. Being behind
    his own side has neither weakness.

    That relative test needs a companion, because an assistant referee beyond
    the last defender scores just as highly on it, and so occasionally does a
    full-back. The touchline test disposes of the assistant, and the keeper's
    lateral position disposes of the full-back: a keeper sits on the midline
    because that is where the goal is.

    The touchline test comes first throughout, because an assistant patrolling
    the corner is close to a goal line too; what distinguishes them is that
    they stay at the edge of the pitch while the keeper stands in the middle
    of it.
    """
    if wears_a_team_kit:
        return PitchRole.OUTFIELD

    shape = footprint(points, length, width)
    if shape is None:
        return PitchRole.UNKNOWN
    if shape.on_touchline >= TOUCHLINE_FRACTION:
        return PitchRole.ASSISTANT
    if shape.in_goal_area >= GOAL_FRACTION:
        return PitchRole.GOALKEEPER
    if (behind_team >= DEEPEST_FRACTION
            and shape.centre_offset <= KEEPER_CENTRE_BAND_M):
        return PitchRole.GOALKEEPER
    return PitchRole.REFEREE


def belongs_in_a_squad(role: PitchRole) -> bool:
    """Whether this track should be given a squad number.

    A goalkeeper is one of the eleven. Officials are not, and an unreadable
    footprint is not either -- leaving a track unnumbered costs a box on the
    overlay, whereas seating an official costs a player their number.
    """
    return role in (PitchRole.OUTFIELD, PitchRole.GOALKEEPER)
