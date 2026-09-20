"""Put every track into a fixed squad of numbered shirts, or into none.

The tracker hands out a new id whenever it is unsure, so a clip of 22 people
ends up with 27 ids and nobody downstream can say how many players there are.
A roster inverts that: the ids exist before the footage does -- 1 to 11 for one
team, 12 to 22 for the other, 0 for the ball -- and every track is placed in
one of them or in none at all. An id then means a person for the whole clip,
and running out of ids is visible rather than silent.

Three rules decide a placement, in order:

  A slot belongs to one team. Colour separates the two kits cleanly, so a
  white shirt can never be given a maroon shirt's number however the boxes
  moved. This is the constraint that stops a crossing swap, and it is applied
  before anything geometric rather than as a repair afterwards.

  A slot holds one person at a time. Two tracks that overlap in time are two
  people and must take different slots, whatever else they have in common.

  A slot prefers whoever it just had. When a track ends and another starts
  soon after and nearby, they are almost always the same player picked back
  up, so the slot is reused and the number survives the gap.

What this cannot do is tell two team-mates apart. Colour gives the team, not
which of the eleven, and at this camera distance a shirt number is a handful
of pixels. Two players in the same kit can still exchange slots, and nothing
here will notice. That limit is the camera's, not the code's.

Referees are deliberately unnumbered: they are not part of either squad, and
giving them ids invites them into team statistics.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from tracking.models import FrameTracks, Track
from utils.logger import get_logger

logger = get_logger("tracking.roster")

Point = Tuple[float, float]

BALL_ID = 0            # the ball is one object and keeps one number
UNASSIGNED_ID = -1     # a referee, or a track the squad had no room for
SLOTS_PER_TEAM = 11


@dataclass
class TrackSpan:
    """One track's lifetime, reduced to what a roster placement needs."""

    track_id: int
    team: int
    first_frame: int
    last_frame: int
    first_point: Point
    last_point: Point


@dataclass
class RosterResult:
    frames: List[FrameTracks]
    assignment: Dict[int, int]        # track_id -> roster id
    overflow: List[int]               # track ids the squad had no room for
    slots_used: Dict[int, int]        # team -> how many of its slots were filled


def foot(bbox: Tuple[float, float, float, float]) -> Point:
    return ((bbox[0] + bbox[2]) / 2.0, bbox[3])


def build_spans(
    frames: Sequence[FrameTracks],
    team_of: Dict[int, int],
    ball_class_name: str = "ball",
) -> List[TrackSpan]:
    """Collapse each track to a span, keeping only tracks with a known team."""
    first: Dict[int, Tuple[int, Point]] = {}
    last: Dict[int, Tuple[int, Point]] = {}
    for frame in sorted(frames, key=lambda f: f.frame_index):
        for track in frame.tracks:
            if track.class_name == ball_class_name:
                continue
            if track.track_id not in team_of:
                continue
            point = foot(track.bbox)
            first.setdefault(track.track_id, (frame.frame_index, point))
            last[track.track_id] = (frame.frame_index, point)
    return [
        TrackSpan(track_id=tid, team=team_of[tid],
                  first_frame=first[tid][0], last_frame=last[tid][0],
                  first_point=first[tid][1], last_point=last[tid][1])
        for tid in sorted(first)
    ]


def assign_slots(
    spans: Sequence[TrackSpan],
    slots_per_team: int = SLOTS_PER_TEAM,
    max_gap_frames: int = 75,
    distance_gate_px: float = 250.0,
) -> Tuple[Dict[int, int], List[int]]:
    """Place each span in a squad number; return ``(assignment, overflow)``.

    Spans are taken in start order so that a slot's history is known by the
    time the next claimant arrives. Among the slots free at that moment, the
    one whose previous occupant left most recently and nearest wins, because
    that is the one most likely to be the same player; a slot nobody has used
    is taken only when no such continuation exists, which keeps the squad
    numbers as few and as long-lived as possible.
    """
    assignment: Dict[int, int] = {}
    overflow: List[int] = []
    # team -> slot index -> (frame it freed up, where it was, was it ever used)
    history: Dict[int, Dict[int, Optional[Tuple[int, Point]]]] = {}
    busy: Dict[int, List[Tuple[int, int]]] = {}     # team -> heap of (end, slot)

    for span in sorted(spans, key=lambda s: (s.first_frame, s.track_id)):
        team = span.team
        slots = history.setdefault(
            team, {i: None for i in range(slots_per_team)})
        occupied = busy.setdefault(team, [])

        # Release every slot whose occupant finished before this span began.
        while occupied and occupied[0][0] < span.first_frame:
            heapq.heappop(occupied)
        taken = {slot for _, slot in occupied}
        free = [s for s in slots if s not in taken]
        if not free:
            overflow.append(span.track_id)
            continue

        def continuity(slot: int) -> Tuple[int, float]:
            """Lower is better; a never-used slot sorts last."""
            prev = slots[slot]
            if prev is None:
                return (1, 0.0)
            end_frame, end_point = prev
            gap = span.first_frame - end_frame
            distance = math.dist(end_point, span.first_point)
            if gap > max_gap_frames or distance > distance_gate_px:
                return (2, distance)      # a stranger: worse than an empty slot
            return (0, distance)

        slot = min(free, key=lambda s: (continuity(s), s))
        assignment[span.track_id] = _roster_id(team, slot, slots_per_team)
        slots[slot] = (span.last_frame, span.last_point)
        heapq.heappush(occupied, (span.last_frame, slot))

    return assignment, overflow


def _roster_id(team: int, slot: int, slots_per_team: int) -> int:
    """Squad numbers run 1..11 for the first team, 12..22 for the second."""
    return team * slots_per_team + slot + 1


def apply_roster(
    frames: Sequence[FrameTracks],
    assignment: Dict[int, int],
    ball_class_name: str = "ball",
    unnumbered_classes: Sequence[str] = ("referee",),
) -> List[FrameTracks]:
    """Rewrite every track's id to its squad number, the ball's, or nothing."""
    out: List[FrameTracks] = []
    for frame in sorted(frames, key=lambda f: f.frame_index):
        tracks: List[Track] = []
        for track in frame.tracks:
            if track.class_name == ball_class_name:
                new_id = BALL_ID
            elif track.class_name in unnumbered_classes:
                new_id = UNASSIGNED_ID
            else:
                new_id = assignment.get(track.track_id, UNASSIGNED_ID)
            tracks.append(track if new_id == track.track_id
                          else Track(frame_id=track.frame_id, track_id=new_id,
                                     class_id=track.class_id,
                                     class_name=track.class_name,
                                     confidence=track.confidence,
                                     bbox=track.bbox,
                                     source_detection_id=track.source_detection_id))
        out.append(FrameTracks(frame_index=frame.frame_index, tracks=tracks))
    return out


def apply(
    frames: Sequence[FrameTracks],
    team_of: Dict[int, int],
    slots_per_team: int = SLOTS_PER_TEAM,
    max_gap_frames: int = 75,
    distance_gate_px: float = 250.0,
    ball_class_name: str = "ball",
    unnumbered_classes: Sequence[str] = ("referee",),
) -> RosterResult:
    """Build the spans, place them, and rewrite the frames."""
    spans = build_spans(frames, team_of, ball_class_name)
    assignment, overflow = assign_slots(
        spans, slots_per_team, max_gap_frames, distance_gate_px)
    filled: Dict[int, set] = {}
    for track_id, roster_id in assignment.items():
        filled.setdefault(team_of[track_id], set()).add(roster_id)
    used = {team: len(ids) for team, ids in sorted(filled.items())}
    logger.info("Roster: %d track(s) placed, %d without a slot, slots used %s",
                len(assignment), len(overflow), used)
    return RosterResult(
        frames=apply_roster(frames, assignment, ball_class_name,
                            unnumbered_classes),
        assignment=assignment, overflow=overflow, slots_used=used)


def overflow_frames(
    frames: Sequence[FrameTracks],
    overflow: Sequence[int],
) -> List[int]:
    """Frames where the squad was short a number -- where to look, not a count.

    Worth surfacing: it is the detector finding a twelfth player in an
    eleven-player team, so the frames it names are where a false positive or a
    doubled box lives.
    """
    wanted = set(overflow)
    return [f.frame_index for f in sorted(frames, key=lambda x: x.frame_index)
            if any(t.track_id in wanted for t in f.tracks)]
