"""Measure how good a set of tracks actually is, not whether the code ran.

The unit tests answer "does this function do what it was told". They cannot
answer "does the system see football correctly", because they feed it blank
frames and a fake model. These are the measurements that do, and they are kept
here rather than inside a test so the same numbers can be printed during a run,
compared between two settings, and asserted against a baseline.

Three of them, deliberately different in what they are blind to:

  mid_pitch_births -- a fresh id away from the frame edge means the tracker
      lost somebody and started over. An id born at the edge is a player
      walking into shot and is not a failure.

  impossible_links -- a step inside one track that no human could have taken.
      These are two different players welded together, which is worse than a
      lost id: it invents distance rather than merely missing it.

  kit_changepoints -- a track whose shirt changes colour mid-life has been
      handed somebody else's boxes. This is the only one of the three that
      catches a crossing swap, where both tracks stay alive and both paths
      stay smooth, so nothing positional moves. The first two report near-zero
      on clips that are visibly full of swaps; this one finds them.

Use all three. A change that improves one while quietly wrecking another is the
normal way a tracking "fix" goes wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]

# A player at this camera scale cannot cover more than this per frame (~10 m/s).
MAX_PX_PER_FRAME = 11.0
# How far from the frame edge an id must appear before its birth counts.
EDGE_FRACTION_X = 0.10
EDGE_FRACTION_Y = 0.12


def foot(bbox: Box) -> Point:
    """Where the player meets the ground -- the only part that is on the pitch."""
    return ((bbox[0] + bbox[2]) / 2.0, bbox[3])


def mid_pitch_births(
    first_seen: Dict[int, Tuple[int, Box]],
    frame_width: int,
    frame_height: int,
    start_frame: int,
) -> List[int]:
    """Ids that appear mid-pitch after the clip has begun."""
    out: List[int] = []
    for track_id, (frame, bbox) in sorted(first_seen.items()):
        if frame <= start_frame:
            continue
        cx, cy = foot(bbox)
        at_edge = (cx < frame_width * EDGE_FRACTION_X
                   or cx > frame_width * (1.0 - EDGE_FRACTION_X)
                   or cy < frame_height * EDGE_FRACTION_Y
                   or cy > frame_height * (1.0 - EDGE_FRACTION_Y))
        if not at_edge:
            out.append(track_id)
    return out


def impossible_links(
    positions: Dict[int, Dict[int, Point]],
    max_px_per_frame: float = MAX_PX_PER_FRAME,
) -> List[Tuple[int, int, int, float]]:
    """``(track_id, from_frame, to_frame, px_per_frame)`` for each unrunnable step.

    Rated per frame, not per step, so a link across a gap is judged by the
    speed it implies rather than punished for the gap's length.
    """
    out: List[Tuple[int, int, int, float]] = []
    for track_id, series in sorted(positions.items()):
        frames = sorted(series)
        for a, b in zip(frames, frames[1:]):
            rate = math.dist(series[a], series[b]) / max(b - a, 1)
            if rate > max_px_per_frame:
                out.append((track_id, a, b, rate))
    return out


@dataclass
class Changepoint:
    """A frame at which a track's appearance stops matching what it was."""

    frame: int
    strength: float


def _mean(colours: Sequence[Sequence[float]]) -> List[float]:
    n = len(colours)
    return [sum(c[i] for c in colours) / n for i in range(len(colours[0]))]


def _scatter(colours: Sequence[Sequence[float]]) -> float:
    """Sum of squared distances to the mean -- additive across segments."""
    m = _mean(colours)
    return sum(math.dist(c, m) ** 2 for c in colours)


def _split_points(
    colours: Sequence[Sequence[float]],
    min_segment: int,
    min_explained: float,
    offset: int,
    out: List[int],
) -> None:
    """Cut indices, chosen by how much scatter each split removes.

    Selection deliberately does NOT ask whether either half is a clean single
    kit, because on a track swapped twice neither half is: white-maroon-white
    splits into "white" and "half and half" whichever way you cut it, and a
    purity test rejects both and finds nothing. Scatter reduction has no such
    blind spot -- it sees that the cut buys a lot of explanation even when what
    remains is still mixed -- and the recursion then resolves the rest.
    Purity is checked afterwards, on the segments that actually come out.
    """
    n = len(colours)
    if n < 2 * min_segment:
        return
    base = _scatter(colours)
    if base <= 0:
        return
    best, best_k = 0.0, None
    for k in range(min_segment, n - min_segment + 1):
        explained = base - _scatter(colours[:k]) - _scatter(colours[k:])
        if explained > best:
            best, best_k = explained, k
    if best_k is None or best / base < min_explained:
        return
    _split_points(colours[:best_k], min_segment, min_explained, offset, out)
    out.append(offset + best_k)
    _split_points(colours[best_k:], min_segment, min_explained,
                  offset + best_k, out)


def kit_changepoints(
    frames: Sequence[int],
    colours: Sequence[Sequence[float]],
    kit_separation: float,
    min_segment: int = 8,
    min_gap: float = 0.55,
    max_spread: float = 0.45,
    min_explained: float = 0.20,
) -> List[Changepoint]:
    """Frames at which this track's shirt genuinely changes kit.

    Counting consecutive samples on the other team is the obvious test and it
    does not work: the smallest boxes carry a shirt patch of a few dozen
    blurred pixels, so their colour rattles between the kits and they flip a
    dozen times, while a real swap flips once and stays.

    So the series is cut where cutting explains the most scatter, and only then
    is each surviving boundary asked the question that matters -- are the two
    stretches either side really a kit apart (``min_gap``) and each tight
    enough to be one shirt (``max_spread``). Flicker never reaches that
    question, because splitting it explains nothing.

    ``kit_separation`` is the distance between the two kits' colours, so the
    thresholds mean the same thing whatever the strips are.
    """
    n = len(colours)
    if kit_separation <= 0 or n < 2 * min_segment or len(frames) != n:
        return []

    cuts: List[int] = []
    _split_points(colours, min_segment, min_explained, 0, cuts)
    if not cuts:
        return []

    bounds = [0] + cuts + [n]
    out: List[Changepoint] = []
    for i, k in enumerate(cuts):
        left = colours[bounds[i]:k]
        right = colours[k:bounds[i + 2]]
        mean_l, mean_r = _mean(left), _mean(right)
        gap = math.dist(mean_l, mean_r) / kit_separation
        if gap < min_gap:
            continue
        spread = max(
            sum(math.dist(c, mean_l) for c in left) / len(left),
            sum(math.dist(c, mean_r) for c in right) / len(right),
        ) / kit_separation
        if spread > max_spread:
            continue
        out.append(Changepoint(frame=frames[k], strength=gap - spread))
    return out
