"""Is a homography actually CORRECT? -- scored against the painted lines.

:meth:`~calibration.homography.Homography.reprojection_error` measures how well
a homography fits the points it was solved from, which is not the same question.
A homography has 8 degrees of freedom, so 5-6 correspondences make the fit an
interpolation: the residual is near zero whatever the points meant. Measured on
this project's own clips, the frames with the *lowest* reprojection error
(0.04 m) were the ones projecting players 40 m from where they stood.

This module scores a homography against evidence it was never fitted to: the
white paint actually visible in the frame.

    1. :func:`white_line_mask` finds the pitch markings with classical CV --
       bright, low-saturation pixels sitting on grass. It knows *where* paint
       is, never *which* line it is, so it cannot inherit the keypoint model's
       mistake (labelling the centre circle as a penalty arc).
    2. :func:`line_alignment_score` projects the pitch template into the image
       with the candidate homography and returns the fraction of it that lands
       on real paint.

A correct homography draws its lines over the painted ones and scores high; a
wrong one draws them over empty grass and scores near zero.

The scores are bimodal in practice, which is what makes them usable as a gate:
accurate frames land around 0.85-0.95, roughly-right ones around 0.50, and
wrong ones below 0.35. :data:`SCORE_GOOD` / :data:`SCORE_USABLE` name those
cuts, but they are defaults to be re-calibrated per camera against hand-clicked
keyframes -- see :func:`suggest_review_segments`, which exists so a human is
asked once per *segment* the automatic path failed on, never per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration.pitch_model import PitchDimensions
from utils.logger import get_logger

logger = get_logger("calibration.quality")

Point = Tuple[float, float]

# A frame at/above this is accurate enough to anchor analytics.
SCORE_GOOD = 0.70
# Below this the homography is wrong; between the two it is roughly right
# (metres of error) -- fine for a minimap, not for speed/distance.
SCORE_USABLE = 0.45

# The projected template must put at least this many sample points inside the
# image before a score means anything.
MIN_SAMPLES = 60

# ...and those samples must be SPREAD OUT, not piled up. A near-singular
# homography collapses the whole template onto a pencil of lines through one
# point; if that point happens to sit on the centre circle, thousands of
# samples land on real paint and the score comes out near 1.0 for a matrix
# that is geometrically meaningless. Counting samples cannot catch this --
# there are plenty of them -- so we also measure how much of the image they
# actually touch, on a coarse grid.
#
# Measured on input_video2: a collapsed homography scoring 0.91 touched 8% of
# the grid, while genuine (even wrong) ones touched 17% and up.
SPREAD_CELL_PX = 64
MIN_SPREAD_FRACTION = 0.12


# ---------------------------------------------------------------------------
# Pitch template as dense polylines (metres)
# ---------------------------------------------------------------------------
def pitch_polylines(d: PitchDimensions = PitchDimensions()) -> List[np.ndarray]:
    """The painted markings as ``(N, 2)`` float arrays in field metres."""
    L, W = d.length, d.width
    cx, cy = L / 2.0, W / 2.0
    pa, ga = d.penalty_area_depth, d.goal_area_depth
    pa_t, pa_b = cy - d.penalty_area_width / 2.0, cy + d.penalty_area_width / 2.0
    ga_t, ga_b = cy - d.goal_area_width / 2.0, cy + d.goal_area_width / 2.0
    r = d.center_circle_radius
    out: List[np.ndarray] = []

    def seg(*pts: Point) -> None:
        out.append(np.array(pts, dtype=np.float64))

    seg((0, 0), (L, 0))                                    # touchlines
    seg((0, W), (L, W))
    seg((0, 0), (0, W))                                    # goal lines
    seg((L, 0), (L, W))
    seg((cx, 0), (cx, W))                                  # halfway
    seg((0, pa_t), (pa, pa_t), (pa, pa_b), (0, pa_b))      # penalty areas
    seg((L, pa_t), (L - pa, pa_t), (L - pa, pa_b), (L, pa_b))
    seg((0, ga_t), (ga, ga_t), (ga, ga_b), (0, ga_b))      # goal areas
    seg((L, ga_t), (L - ga, ga_t), (L - ga, ga_b), (L, ga_b))

    theta = np.linspace(0, 2 * np.pi, 181)                 # centre circle
    out.append(np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1))

    # Penalty arcs: the part of the circle outside the penalty-area line.
    dx = abs(pa - d.penalty_spot_distance)
    half = np.arccos(min(dx / r, 1.0))
    ang = np.linspace(-half, half, 61)
    for spot_x, sign in ((d.penalty_spot_distance, +1.0),
                         (L - d.penalty_spot_distance, -1.0)):
        out.append(np.stack(
            [spot_x + sign * r * np.cos(ang), cy + r * np.sin(ang)], axis=1))
    return out


def densify(poly: np.ndarray, per_metre: float = 2.0) -> np.ndarray:
    """Resample a polyline so long straight runs are sampled evenly."""
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 2:
        return poly
    parts = []
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        n = max(int(float(np.hypot(*(b - a))) * per_metre), 2)
        parts.append(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None])
    return np.vstack(parts)


# ---------------------------------------------------------------------------
# Where is the paint?
# ---------------------------------------------------------------------------
def white_line_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/1) of the painted pitch markings.

    Deliberately classical: paint is what is locally brighter than the grass
    around it and washed-out in colour, restricted to the grass region so the
    crowd, hoardings and the scoreboard cannot contribute. Nothing here knows
    which line is which, which is exactly why it is trustworthy as an
    independent check on a model that does.
    """
    if frame.ndim != 3:
        raise ValueError("frame must be a BGR image")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    grass = ((hue > 25) & (hue < 95) & (sat > 40)).astype(np.uint8)
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    # Local contrast against a heavily blurred copy: robust to floodlights,
    # mown stripes and a night match's colour cast.
    blurred = cv2.GaussianBlur(val, (0, 0), 12)
    brighter = (val.astype(np.int16) - blurred.astype(np.int16)) > 12

    lines = (brighter & (sat < 110) & grass.astype(bool)).astype(np.uint8)
    return cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def project_field_to_image(
    inverse: np.ndarray, points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project field points to image pixels; returns ``(xy, valid)``.

    Points whose homogeneous ``w`` flips sign fell behind the camera (the
    projection wraps around the horizon) and are marked invalid rather than
    drawn on the far side of the image.
    """
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ inverse.T
    w = homogeneous[:, 2]
    valid = np.abs(w) > 1e-9
    xy = np.full((len(points), 2), np.nan)
    xy[valid] = homogeneous[valid, :2] / w[valid, None]
    if valid.any():
        valid &= np.sign(w) == np.sign(w[valid][0])
    valid &= np.all(np.isfinite(xy), axis=1)
    return xy, valid


def line_alignment_score(
    frame: np.ndarray,
    homography,
    tolerance_px: float = 6.0,
    mask: Optional[np.ndarray] = None,
    dims: PitchDimensions = PitchDimensions(),
) -> Tuple[Optional[float], int]:
    """Fraction of the projected pitch template that lands on real paint.

    Returns ``(score, n_samples)``. ``score`` is ``None`` when the frame has
    too little visible paint, or when the homography puts too little of the
    template inside the image, to judge either way -- an honest "unknown"
    rather than a misleading number.

    A perfect homography does not score 1.0: parts of the template are hidden
    behind players or fall outside the visible pitch. Read the score against
    :data:`SCORE_GOOD` / :data:`SCORE_USABLE`, not against 100%.
    """
    if mask is None:
        mask = white_line_mask(frame)
    if int(mask.sum()) < 200:
        return None, 0

    # Distance from every pixel to the nearest painted pixel, so scoring a
    # sample is one lookup instead of a search.
    distance = cv2.distanceTransform(
        (1 - mask).astype(np.uint8), cv2.DIST_L2, 3)

    height, width = frame.shape[:2]
    inverse = np.asarray(homography.inverse, dtype=np.float64)
    hits = total = 0
    touched: set = set()
    for poly in pitch_polylines(dims):
        pts = densify(poly)
        xy, valid = project_field_to_image(inverse, pts)
        inside = (
            valid
            & (xy[:, 0] >= 0) & (xy[:, 0] < width)
            & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        )
        if not inside.any():
            continue
        cols = xy[inside, 0].astype(np.int32)
        rows = xy[inside, 1].astype(np.int32)
        total += int(inside.sum())
        hits += int((distance[rows, cols] <= tolerance_px).sum())
        touched.update(
            zip((cols // SPREAD_CELL_PX).tolist(),
                (rows // SPREAD_CELL_PX).tolist()))

    if total < MIN_SAMPLES:
        return None, total
    if not _spread_enough(touched, width, height):
        # Collapsed onto a pencil of lines: plenty of samples, all in one
        # place. Report "unknown" rather than the flattering number, so a
        # degenerate matrix can never become an anchor.
        return None, total
    return hits / total, total


def _spread_enough(touched: set, width: int, height: int) -> bool:
    """Do the projected samples touch enough of the image to be a real fit?"""
    grid_cells = (width // SPREAD_CELL_PX + 1) * (height // SPREAD_CELL_PX + 1)
    return len(touched) >= MIN_SPREAD_FRACTION * grid_cells


def classify_score(score: Optional[float]) -> str:
    """``"good"`` / ``"usable"`` / ``"wrong"`` / ``"unknown"``."""
    if score is None:
        return "unknown"
    if score >= SCORE_GOOD:
        return "good"
    if score >= SCORE_USABLE:
        return "usable"
    return "wrong"


# ---------------------------------------------------------------------------
# Segments: ask a human once per shot, not once per frame
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """A run of frames the same calibration can serve.

    ``anchor_frame`` is the best-scoring frame in the run. When it is ``None``
    the automatic path failed across the whole segment and this is the one
    place a human is worth asking -- one keyframe, not one per frame.
    """

    start: int
    end: int
    anchor_frame: Optional[int] = None
    anchor_score: Optional[float] = None

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def needs_review(self) -> bool:
        return self.anchor_frame is None

    def contains(self, frame: int) -> bool:
        return self.start <= int(frame) <= self.end


def split_segments(
    frame_indices: Sequence[int], cut_frames: Sequence[int] = ()
) -> List[Tuple[int, int]]:
    """Split a frame range into ``(start, end)`` runs at the given cuts.

    ``cut_frames`` are the first frames of a new shot. Cuts outside the range
    are ignored, so a shot detector may over-report without breaking this.
    """
    frames = sorted(int(f) for f in frame_indices)
    if not frames:
        return []
    cuts = sorted({int(c) for c in cut_frames
                   if frames[0] < int(c) <= frames[-1]})
    bounds: List[Tuple[int, int]] = []
    start = frames[0]
    for cut in cuts:
        bounds.append((start, cut - 1))
        start = cut
    bounds.append((start, frames[-1]))
    return [(a, b) for a, b in bounds if b >= a]


def suggest_review_segments(
    scores: Dict[int, Optional[float]],
    cut_frames: Sequence[int] = (),
    good_score: float = SCORE_GOOD,
) -> List[Segment]:
    """Turn per-frame scores into segments, each with its best anchor.

    This is what keeps the human out of the loop for most of a video: a
    segment holding even one well-scoring frame is anchored automatically (the
    anchor propagates across the segment via camera motion), and only a
    segment with none is flagged for a single manual keyframe.
    """
    if not scores:
        return []
    segments: List[Segment] = []
    for start, end in split_segments(list(scores), cut_frames):
        best_frame: Optional[int] = None
        best_score: Optional[float] = None
        for frame, score in scores.items():
            if score is None or not (start <= frame <= end):
                continue
            if score >= good_score and (best_score is None or score > best_score):
                best_frame, best_score = int(frame), float(score)
        segments.append(Segment(start, end, best_frame, best_score))
    return segments


def review_effort(segments: Sequence[Segment]) -> dict:
    """How much human work a video actually needs, and what it buys."""
    total = len(segments)
    review = [s for s in segments if s.needs_review]
    frames = sum(s.length for s in segments) or 1
    covered = sum(s.length for s in segments if not s.needs_review)
    return {
        "segments": total,
        "auto_anchored": total - len(review),
        "need_manual_keyframe": len(review),
        "frames_total": frames,
        "frames_auto_covered": covered,
        "auto_coverage": covered / frames,
        "review_frames": [s.start for s in review],
    }
