"""Self-verifying automatic calibration; a human only for what it can't do.

The automatic path used to accept whatever the keypoint model produced, because
its only gate -- reprojection error -- cannot tell a correct homography from a
confidently wrong one (an 8-DoF fit through 5-6 points is an interpolation).
Measured on this project's clips, roughly three quarters of the accepted frames
were wrong, and the *lowest*-error frames were among the worst.

This module closes that loop with :mod:`calibration.calibration_quality`, which
scores a homography against paint it was never fitted to:

    1. run the keypoint backend on sampled frames,
    2. **score every candidate** and keep only the ones that land on the real
       lines -- the rest are dropped, not trusted,
    3. propagate each surviving anchor across its shot with camera motion,
       **re-scoring as it goes** and stopping where it stops being right, so
       the reach of an anchor is measured per video instead of assumed,
    4. hand back a provider that knows which frames it cannot serve, so
       analytics can skip them instead of inventing positions,
    5. report the segments nothing covered -- the only place a human is asked,
       one keyframe per segment, never per frame.

The decision logic is pure and unit-tested with injected fakes; only
:func:`detect_shot_cuts` and :func:`calibrate_video` touch video or models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from calibration.calibration_quality import (
    SCORE_GOOD,
    SCORE_USABLE,
    Segment,
    review_effort,
    suggest_review_segments,
)
from utils.logger import get_logger

logger = get_logger("calibration.auto")

# How far an anchor may be carried before we stop even if it still scores well
# (a guard against a long, slowly-drifting run, not the primary limit -- the
# score is).
MAX_PROPAGATION_FRAMES = 250


# ---------------------------------------------------------------------------
# Shot boundaries
# ---------------------------------------------------------------------------
def detect_shot_cuts(
    video_path: str | Path, threshold: float = 0.45, max_frames: Optional[int] = None
) -> List[int]:
    """Frames that begin a new shot, from a colour-histogram jump.

    A broadcast cut to a replay or a close-up invalidates the calibration
    entirely, and nearest-in-time anchor selection would otherwise happily
    carry a homography straight across one.
    """
    import cv2

    from utils.video_io import VideoReader

    cuts: List[int] = []
    previous = None
    with VideoReader(video_path) as reader:
        for index, frame in reader.frames():
            small = cv2.resize(frame, (160, 90))
            hist = cv2.calcHist(
                [cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None,
                [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if previous is not None:
                distance = 1.0 - cv2.compareHist(previous, hist, cv2.HISTCMP_CORREL)
                if distance > threshold:
                    cuts.append(index)
            previous = hist
            if max_frames is not None and index >= max_frames:
                break
    logger.info("Detected %d shot cut(s)", len(cuts))
    return cuts


# ---------------------------------------------------------------------------
# Anchor selection (pure)
# ---------------------------------------------------------------------------
def select_anchors(
    scores: Dict[int, Optional[float]], min_score: float = SCORE_GOOD
) -> List[int]:
    """Frames whose homography actually landed on the painted lines."""
    return sorted(
        f for f, s in scores.items() if s is not None and s >= min_score)


def _segment_for(frame: int, segments: Sequence[Segment]) -> Optional[Segment]:
    for segment in segments:
        if segment.contains(frame):
            return segment
    return None


# ---------------------------------------------------------------------------
# Propagation that checks itself (pure; verification is injected)
# ---------------------------------------------------------------------------
def propagate_verified(
    anchors: Dict[int, np.ndarray],
    transforms: Dict[int, np.ndarray],
    segments: Sequence[Segment],
    verify: Callable[[int, np.ndarray], Optional[float]],
    min_score: float = SCORE_USABLE,
    max_reach: int = MAX_PROPAGATION_FRAMES,
) -> Dict[int, Tuple[np.ndarray, float]]:
    """Carry each anchor across its shot, keeping only what still verifies.

    ``anchors`` maps a frame to its image->field matrix, ``transforms`` maps
    frame ``k`` to the background motion ``k-1 -> k``, and ``verify`` scores a
    propagated matrix on a frame (``None`` when it cannot be judged).

    Walking outward from an anchor:  forward ``H_k = H_{k-1} @ inv(M_k)``,
    backward ``H_{k-1} = H_k @ M_k``.  The walk stops at the segment edge, at
    ``max_reach``, at a missing transform, or -- the point of this function --
    as soon as the propagated matrix no longer scores at least ``min_score``.
    An unjudgeable frame is carried but not counted as progress, so a patch of
    frames with no visible paint does not end an otherwise good run.

    Returns ``{frame: (matrix, score)}``, keeping the best-scoring candidate
    when two anchors reach the same frame.
    """
    best: Dict[int, Tuple[np.ndarray, float]] = {}

    def offer(frame: int, matrix: np.ndarray, score: float) -> None:
        current = best.get(frame)
        if current is None or score > current[1]:
            best[frame] = (matrix, score)

    for anchor_frame in sorted(anchors):
        anchor = np.asarray(anchors[anchor_frame], dtype=np.float64)
        segment = _segment_for(anchor_frame, segments)
        lo = segment.start if segment else anchor_frame
        hi = segment.end if segment else anchor_frame
        anchor_score = verify(anchor_frame, anchor)
        offer(anchor_frame, anchor, anchor_score if anchor_score is not None else 1.0)

        for direction in (+1, -1):
            current = anchor.copy()
            frame = anchor_frame
            for _step in range(max_reach):
                nxt = frame + direction
                if nxt < lo or nxt > hi:
                    break
                # Forward needs M_{k}; backward needs M_{k+1}.
                motion = transforms.get(nxt if direction > 0 else frame)
                if motion is None:
                    break
                try:
                    current = (current @ np.linalg.inv(motion) if direction > 0
                               else current @ motion)
                except np.linalg.LinAlgError:
                    break
                if not np.all(np.isfinite(current)):
                    break
                score = verify(nxt, current)
                if score is None:
                    frame = nxt          # cannot judge: carry on, claim nothing
                    continue
                if score < min_score:
                    break                # the anchor has run out of reach
                offer(nxt, current.copy(), score)
                frame = nxt
    return best


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------
@dataclass
class CalibrationCoverage:
    """What the automatic pass managed, and what it wants a human for."""

    frames_total: int = 0
    frames_covered: int = 0
    anchors: int = 0
    segments: List[Segment] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.frames_covered / self.frames_total if self.frames_total else 0.0

    @property
    def review_frames(self) -> List[int]:
        """One frame per segment the automatic path could not serve."""
        return [s.start for s in self.segments if s.needs_review]

    def summary(self) -> dict:
        effort = review_effort(self.segments)
        return {
            "frames_total": self.frames_total,
            "frames_covered": self.frames_covered,
            "coverage": round(self.coverage, 4),
            "anchors": self.anchors,
            "segments": effort["segments"],
            "need_manual_keyframe": effort["need_manual_keyframe"],
            "review_frames": self.review_frames,
        }


class VerifiedHomographyProvider:
    """Per-frame homographies that were each checked against the real lines.

    Drop-in for the other providers (``for_frame``), with one addition that
    matters more than the interface: :meth:`is_valid`. A frame with no verified
    calibration returns ``False``, so a caller can leave that frame out instead
    of projecting a player onto a fabricated spot -- which is what the current
    pipeline does when it clamps an impossible projection onto the touchline.
    """

    def __init__(
        self,
        verified: Dict[int, Tuple[np.ndarray, float]],
        coverage: Optional[CalibrationCoverage] = None,
    ) -> None:
        from calibration.homography import Homography

        if not verified:
            raise ValueError("no frame passed verification")
        self._frames = sorted(verified)
        self._homographies = {
            f: Homography(np.asarray(m, dtype=np.float64))
            for f, (m, _s) in verified.items()
        }
        self._scores = {f: float(s) for f, (_m, s) in verified.items()}
        self.coverage = coverage or CalibrationCoverage(
            frames_total=len(verified), frames_covered=len(verified))

    def is_valid(self, frame_index: int) -> bool:
        """True when this exact frame has a verified calibration."""
        return int(frame_index) in self._homographies

    def score(self, frame_index: int) -> Optional[float]:
        return self._scores.get(int(frame_index))

    def for_frame(self, frame_index: int):
        index = int(frame_index)
        homography = self._homographies.get(index)
        if homography is not None:
            return homography
        nearest = min(self._frames, key=lambda f: abs(f - index))
        return self._homographies[nearest]

    @property
    def frame_numbers(self) -> List[int]:
        return list(self._frames)

    def mean_score(self) -> float:
        return float(np.mean(list(self._scores.values())))

    def reprojection_error(self) -> float:
        """Kept for interface compatibility; the score is the real metric.

        Deliberately not a reprojection error: this provider's frames come from
        propagation and carry no correspondences of their own, and the metric
        was shown not to track correctness anyway.
        """
        return float("nan")

    def __len__(self) -> int:
        return len(self._homographies)


# ---------------------------------------------------------------------------
# Assembly (pure)
# ---------------------------------------------------------------------------
def build_verified_provider(
    candidate_matrices: Dict[int, np.ndarray],
    scores: Dict[int, Optional[float]],
    transforms: Dict[int, np.ndarray],
    frame_indices: Sequence[int],
    verify: Callable[[int, np.ndarray], Optional[float]],
    cut_frames: Sequence[int] = (),
    min_anchor_score: float = SCORE_GOOD,
    min_keep_score: float = SCORE_USABLE,
) -> Tuple[Optional[VerifiedHomographyProvider], CalibrationCoverage]:
    """Scored candidates + camera motion -> a verified provider + a coverage report.

    Returns ``(None, coverage)`` when nothing verified, in which case every
    segment is flagged for review -- the honest outcome for a video the
    automatic path cannot handle, instead of a provider full of wrong matrices.
    """
    frames = sorted(int(f) for f in frame_indices)
    segments = suggest_review_segments(scores, cut_frames, min_anchor_score)
    anchor_frames = select_anchors(scores, min_anchor_score)
    anchors = {f: np.asarray(candidate_matrices[f], dtype=np.float64)
               for f in anchor_frames if f in candidate_matrices}

    coverage = CalibrationCoverage(
        frames_total=len(frames), anchors=len(anchors), segments=list(segments))
    if not anchors:
        logger.warning(
            "No frame passed verification (>= %.2f); every segment needs a "
            "manual keyframe.", min_anchor_score)
        return None, coverage

    verified = propagate_verified(
        anchors, transforms, segments, verify, min_keep_score)
    coverage.frames_covered = len(verified)
    if not verified:
        return None, coverage

    logger.info(
        "Verified calibration: %d anchors -> %d/%d frames (%.0f%%); "
        "%d segment(s) need a manual keyframe",
        len(anchors), len(verified), max(len(frames), 1),
        100.0 * coverage.coverage, len(coverage.review_frames))
    return VerifiedHomographyProvider(verified, coverage), coverage
