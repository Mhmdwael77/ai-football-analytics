"""Appearance-based ID-swap correction (post-tracking).

When two players cross, BoT-SORT can *exchange* their IDs: a track that
was blue becomes red and the other goes red->blue. Stitching can't fix
this (both tracks stay alive the whole time) — it's a concurrent swap, not
a broken/gap track.

This module detects swaps from jersey appearance. Each track is given a
team label per sampled frame; a track whose label *flips* mid-way is
paired with another track that flips the OPPOSITE way at the same time and
place (they crossed), and their identities are swapped back from the
crossing frame onward — so each player keeps one consistent id + team.

Pure helpers (``detect_flip``, ``match_swaps``, ``apply_swaps``) are
unit-testable without any video; :class:`SwapCorrector` wires in the crop
sampling + team clustering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from tracking.models import FrameTracks, Track
from utils.logger import get_logger

logger = get_logger("tracking.swap_corrector")

Point = Tuple[float, float]


@dataclass
class SwapInterval:
    """A stretch where a track's team label disagrees with its dominant team.

    During ``[start, end]`` the track (whose dominant team is ``own_team``)
    looks like ``other_team`` — i.e. it picked up the *other* player there.
    """

    track_id: int
    start: int
    end: int
    own_team: int
    other_team: int
    n_samples: int = 0          # off-team samples in this run (confidence)


# A swap to undo: ids ``a`` and ``b`` are exchanged over ``[start, end]``.
Swap = Tuple[int, int, int, int]
# A split: track ``id`` over ``[start, end]`` is reassigned to ``new_id``.
Split = Tuple[int, int, int, int]


@dataclass
class SwapResult:
    frames: List[FrameTracks]
    swaps: List[Swap]                   # (id_a, id_b, start_frame, end_frame)
    n_swaps: int = 0
    splits: List[Split] = field(default_factory=list)  # (id, start, end, new_id)


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------
def find_swap_intervals(
    timeline: Sequence[Tuple[int, Optional[int]]],
    min_run: int = 2,
) -> "Tuple[Optional[int], List[SwapInterval]]":
    """Find runs where the team label departs from the track's dominant team.

    Handles *temporary* swaps (cross then cross back), not just a single
    permanent flip. Returns ``(dominant_team, [SwapInterval, ...])``.
    """
    labels = [(f, t) for f, t in timeline if t is not None]
    if len(labels) < 2 * min_run:
        return None, []
    teams = [t for _, t in labels]
    dominant = max(set(teams), key=teams.count)

    intervals: List[SwapInterval] = []
    run: List[Tuple[int, int]] = []
    for frame, team in labels:
        if team != dominant:
            run.append((frame, team))
        else:
            if len(run) >= min_run:
                intervals.append(SwapInterval(
                    track_id=-1, start=run[0][0], end=run[-1][0],
                    own_team=dominant, other_team=run[0][1],
                    n_samples=len(run)))
            run = []
    if len(run) >= min_run:
        intervals.append(SwapInterval(
            track_id=-1, start=run[0][0], end=run[-1][0],
            own_team=dominant, other_team=run[0][1], n_samples=len(run)))
    return dominant, intervals


def match_swaps(
    intervals: Sequence[SwapInterval],
    center_at: "callable",
    frame_tol: int = 25,
    distance_gate_px: float = 220.0,
) -> List[Swap]:
    """Pair opposite swap-intervals that overlap in time and place.

    Track A (dominant team X) looking like Y over ``[sa, ea]`` is paired
    with track B (dominant Y) looking like X over an overlapping window,
    if they are physically close there. The exchange is applied over the
    union of the two windows. ``center_at(track_id, frame)`` returns a pixel
    center near a frame (or ``None``).
    """
    ordered = sorted(intervals, key=lambda iv: iv.start)
    used: set = set()
    swaps: List[Swap] = []
    for i, a in enumerate(ordered):
        if id(a) in used:
            continue
        for b in ordered[i + 1:]:
            if id(b) in used:
                continue
            # Opposite directions: A is X->looks-Y, B is Y->looks-X.
            if not (a.own_team == b.other_team and a.other_team == b.own_team):
                continue
            # Time windows must overlap (allowing a small tolerance).
            if max(a.start, b.start) > min(a.end, b.end) + frame_tol:
                continue
            mid = (max(a.start, b.start) + min(a.end, b.end)) // 2
            ca, cb = center_at(a.track_id, mid), center_at(b.track_id, mid)
            if ca is None or cb is None:
                continue
            if math.hypot(ca[0] - cb[0], ca[1] - cb[1]) > distance_gate_px:
                continue
            swaps.append((a.track_id, b.track_id,
                          min(a.start, b.start), max(a.end, b.end)))
            used.add(id(a))
            used.add(id(b))
            break
    return swaps


def apply_swaps(
    frames: Sequence[FrameTracks], swaps: Sequence[Swap]
) -> List[FrameTracks]:
    """Exchange each pair's ids over its ``[start, end]`` window."""
    ordered = sorted(swaps, key=lambda s: s[2])
    out: List[FrameTracks] = []
    for frame in frames:
        perm: Dict[int, int] = {}
        for a, b, start, end in ordered:
            if start <= frame.frame_index <= end:
                pa, pb = perm.get(a, a), perm.get(b, b)
                perm[a], perm[b] = pb, pa
        new_tracks: List[Track] = []
        seen: set = set()
        for track in frame.tracks:
            nid = perm.get(track.track_id, track.track_id)
            if nid in seen:
                continue
            seen.add(nid)
            new_tracks.append(
                track if nid == track.track_id else replace(track, track_id=nid))
        out.append(FrameTracks(frame_index=frame.frame_index, tracks=new_tracks))
    return out


def join_cost(
    pos_a: Dict[int, Point], pos_b: Dict[int, Point], frame: int
) -> Optional[float]:
    """Pixels per frame implied by handing A's boxes to B from ``frame`` on.

    Exchanging at ``frame`` joins A's last observation before it to B's first
    observation from it onward, and vice versa; if either track is missing
    boxes around there, the join spans that gap too. The cost is the worse of
    the two sides, so a cheap join means both players stay where they were.
    """
    worst = 0.0
    for src, dst in ((pos_a, pos_b), (pos_b, pos_a)):
        before = [f for f in src if f < frame]
        after = [f for f in dst if f >= frame]
        if not before or not after:
            return None
        f0, f1 = max(before), min(after)
        worst = max(worst, math.dist(src[f0], dst[f1]) / max(f1 - f0, 1))
    return worst


def _cheapest_frame(
    pos_a: Dict[int, Point], pos_b: Dict[int, Point], frame: int, tol: int
) -> int:
    best, at = None, frame
    for f in range(frame - tol, frame + tol + 1):
        cost = join_cost(pos_a, pos_b, f)
        if cost is not None and (best is None or cost < best):
            best, at = cost, f
    return at


def refine_swap_edges(
    swaps: Sequence[Swap],
    positions: Dict[int, Dict[int, Point]],
    tol: int = 45,
) -> List[Swap]:
    """Move each exchange to the frame where it costs the least motion.

    Appearance says *that* two tracks were exchanged, but it cannot say
    *when*: a jersey only reads as the other team once the two players have
    drawn apart again, which is tens of frames after the boxes were actually
    handed over. Exchanging at that boundary makes each repaired track
    teleport across the gap they had opened -- on one clip this turned a 13
    px/frame worst case into 376, i.e. the fix flung players across the pitch.

    So the window is nudged to the cheapest join nearby, which is the crossing
    itself. The pairing is unchanged; only its edges move.
    """
    out: List[Swap] = []
    for a, b, start, end in swaps:
        pos_a, pos_b = positions.get(a, {}), positions.get(b, {})
        if not pos_a or not pos_b:
            out.append((a, b, start, end))
            continue
        # Only an edge that sits inside both tracks is a crossing that can be
        # moved. A window reaching the start or the end of the tracks means the
        # exchange was already in force, or never undone, and has no edge there
        # -- nudging such a boundary would carve a swap out of a correct
        # stretch rather than align it to anything.
        first = min(min(pos_a), min(pos_b))
        last = max(max(pos_a), max(pos_b))
        if start > first:
            start = _cheapest_frame(pos_a, pos_b, start, tol)
        if end < last:
            end = max(_cheapest_frame(pos_a, pos_b, end + 1, tol) - 1, start)
        out.append((a, b, start, end))
    return out


def unpaired_intervals(
    intervals: Sequence[SwapInterval],
    swaps: Sequence[Swap],
    min_samples: int = 3,
) -> List[SwapInterval]:
    """Swap-intervals left over after pairing -- candidates for a split.

    An interval where a track briefly looked like the other team but that
    pairing couldn't match to a crossing partner is a single-track DRIFT
    (the box jumped to a different player and came back). Such an interval,
    if it is sustained enough (``min_samples``), is split off into its own
    id so that segment is a separate, correctly-coloured track.
    """
    out: List[SwapInterval] = []
    for iv in intervals:
        if iv.n_samples < min_samples:
            continue
        covered = any(
            iv.track_id in (a, b) and not (iv.end < s or iv.start > e)
            for a, b, s, e in swaps
        )
        if not covered:
            out.append(iv)
    return out


def plan_splits(
    intervals: Sequence[SwapInterval], start_id: int
) -> List[Split]:
    """Give each interval's ``[start, end]`` window a fresh, unique id."""
    splits: List[Split] = []
    next_id = start_id
    for iv in sorted(intervals, key=lambda i: (i.track_id, i.start)):
        splits.append((iv.track_id, iv.start, iv.end, next_id))
        next_id += 1
    return splits


def apply_splits(
    frames: Sequence[FrameTracks], splits: Sequence[Split]
) -> List[FrameTracks]:
    """Reassign a track's id to ``new_id`` inside each split's window."""
    out: List[FrameTracks] = []
    for frame in frames:
        new_tracks: List[Track] = []
        for track in frame.tracks:
            nid = track.track_id
            for tid, start, end, new_id in splits:
                if track.track_id == tid and start <= frame.frame_index <= end:
                    nid = new_id
                    break
            new_tracks.append(
                track if nid == track.track_id else replace(track, track_id=nid))
        out.append(FrameTracks(frame_index=frame.frame_index, tracks=new_tracks))
    return out


# ---------------------------------------------------------------------------
# Runner (appearance sampling + team clustering)
# ---------------------------------------------------------------------------
class SwapCorrector:
    """Detects and fixes appearance ID-swaps in stitched tracks."""

    def __init__(
        self,
        ball_class_name: str = "ball",
        samples_per_track: int = 40,
        min_track_length: int = 30,
        frame_tol: int = 25,
        distance_gate_px: float = 220.0,
        min_purity: float = 0.8,
        split_unpaired: bool = True,
        min_split_samples: int = 3,
        min_split_purity: float = 0.7,
        edge_refine_frames: int = 45,
    ) -> None:
        self.ball_class_name = ball_class_name
        self.samples_per_track = samples_per_track
        self.min_track_length = min_track_length
        self.frame_tol = frame_tol
        self.distance_gate_px = distance_gate_px
        self.min_purity = min_purity
        self.split_unpaired = split_unpaired
        self.min_split_samples = min_split_samples
        self.min_split_purity = min_split_purity
        self.edge_refine_frames = edge_refine_frames

    def correct(self, frames: Sequence[FrameTracks], video_path) -> SwapResult:
        """Find swaps from jersey appearance and return corrected frames."""
        from role_refinement.appearance_extractor import (
            AppearanceExtractor,
            build_track_history,
        )
        from role_refinement.role_models import TrackAppearance
        from role_refinement.role_refiner import VideoCropProvider, _invert_plan
        from role_refinement.team_clusterer import TeamClusterer
        from utils.config_loader import RoleRefinementConfig

        rr = RoleRefinementConfig(
            samples_per_track=self.samples_per_track,
            min_track_length=self.min_track_length,
            ball_class_name=self.ball_class_name,
        )
        extractor = AppearanceExtractor(rr)
        history = build_track_history(frames)

        plan = extractor.sample_frames(history)
        requests = _invert_plan(history, plan, rr)
        provider = VideoCropProvider.from_video(video_path, requests)

        # Per-track (frame_id -> vector) timeline, aligned by construction.
        per_track: Dict[int, List[Tuple[int, "object"]]] = {}
        for track_id, observations in history.items():
            if observations and observations[0].class_name == self.ball_class_name:
                continue
            bbox_by_frame = {o.frame_id: o.bbox for o in observations}
            seq = []
            for frame_id in plan.get(track_id, []):
                crop = provider.get(track_id, frame_id, bbox_by_frame.get(frame_id))
                vec = extractor.vector_from_crop(crop)
                if vec is not None:
                    seq.append((frame_id, vec))
            if seq:
                per_track[track_id] = seq

        # Cluster teams from the (stable) track mean vectors.
        import numpy as np

        appearances: Dict[int, TrackAppearance] = {}
        for track_id, seq in per_track.items():
            vecs = np.stack([v for _, v in seq], axis=0)
            appearances[track_id] = TrackAppearance(
                track_id=track_id,
                detected_class="player",
                vector=vecs.mean(axis=0),
                sample_vectors=vecs,
                n_samples=len(vecs),
                track_length=len(history[track_id]),
            )
        cluster = TeamClusterer(rr).cluster(appearances)
        if not cluster.ok:
            logger.info("Swap correction: team clustering unavailable; skipping")
            return SwapResult(frames=list(frames), swaps=[], n_swaps=0)

        # Build per-track team-label timelines and find swap intervals
        # (runs where the label departs from the track's dominant team).
        all_intervals: List[SwapInterval] = []
        purity: Dict[int, float] = {}        # track -> dominant-team fraction
        for track_id, seq in per_track.items():
            timeline = []
            for frame_id, vec in seq:
                idx, dist = cluster.nearest(vec)
                label = idx if (idx >= 0 and dist <= rr.outlier_distance_threshold) \
                    else None
                timeline.append((frame_id, label))
            labelled = [t for _, t in timeline if t is not None]
            if labelled:
                purity[track_id] = max(map(labelled.count, set(labelled))) \
                    / len(labelled)
            _dom, intervals = find_swap_intervals(timeline)
            for interval in intervals:
                interval.track_id = track_id
                all_intervals.append(interval)

        centers = _center_lookup(history)
        swaps = match_swaps(
            all_intervals, centers, self.frame_tol, self.distance_gate_px)
        if swaps:
            swaps = refine_swap_edges(
                swaps, _position_lookup(history), self.edge_refine_frames)
        corrected = apply_swaps(frames, swaps) if swaps else list(frames)

        # Split the drift intervals that had no crossing partner: each becomes
        # its own id, so a box that jumped onto a different player no longer
        # carries that player's frames under the original id. Only split tracks
        # with a CLEAR home team (high purity) -- a ~50/50 track is just a
        # colour-ambiguous single player, and splitting it would fragment it.
        splits: List[Split] = []
        if self.split_unpaired:
            remaining = [
                iv for iv in unpaired_intervals(
                    all_intervals, swaps, self.min_split_samples)
                if purity.get(iv.track_id, 0.0) >= self.min_split_purity
            ]
            if remaining:
                max_id = max(
                    (t.track_id for fr in frames for t in fr.tracks), default=0)
                splits = plan_splits(remaining, start_id=max(max_id + 1, 7000))
                corrected = apply_splits(corrected, splits)

        logger.info(
            "Swap correction: %d interval(s), %d swap(s) fixed, %d drift split(s)",
            len(all_intervals), len(swaps), len(splits))
        return SwapResult(
            frames=corrected, swaps=swaps, n_swaps=len(swaps), splits=splits)


def _position_lookup(history: Dict[int, List[Track]]) -> Dict[int, Dict[int, Point]]:
    """``{track_id: {frame: center}}`` -- exact frames, unlike ``_center_lookup``.

    Placing the exchange needs to know which frames a track actually has, so
    that a join spanning a gap is charged for it.
    """
    return {tid: {o.frame_id: o.center for o in obs}
            for tid, obs in history.items() if obs}


def _center_lookup(history: Dict[int, List[Track]]):
    """Return ``center_at(track_id, frame)`` -> nearest-observation center."""
    table: Dict[int, List[Tuple[int, Point]]] = {}
    for track_id, observations in history.items():
        table[track_id] = [(o.frame_id, o.center) for o in observations]

    def center_at(track_id: int, frame: int) -> Optional[Point]:
        seq = table.get(track_id)
        if not seq:
            return None
        return min(seq, key=lambda fc: abs(fc[0] - frame))[1]

    return center_at
