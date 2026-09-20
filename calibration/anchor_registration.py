"""Carry a hand-clicked calibration across a video by DIRECT registration.

The problem this solves
-----------------------
A homography is only trustworthy where it was constrained. Measured on this
project's clips, a calibration clicked on the centre and right of the pitch
was accurate to 4-5 cm there and wrong by **1.5 metres** on the left, which the
clicks never covered. Extrapolation is the single recurring failure mode, and
it shows up at two levels:

* the pitch homography extrapolating into pitch areas nobody clicked, and
* the frame-to-frame transform extrapolating outside the image it was fitted
  in -- carrying a left-side point into a frame that cannot see the left side
  placed it 900 px beyond the edge, and the error came with it.

So this module never extrapolates. It keeps several verified anchors and, for
each frame, uses the one that shares the most image features with it.

Why direct registration rather than chaining
--------------------------------------------
Composing per-frame transforms (``camera_motion``) accumulates error: every
step multiplies, and after ~50 frames the calibration has drifted away. It
also models the step as an affine, which cannot represent what a rotating or
zooming camera does to the image at all. Matching the target frame *straight*
to the anchor pays one error instead of a hundred, and solves a full 8-DoF
homography.

Measured on match3 (60 checkpoints across 1500 frames), from clicks alone:

======================================  =========  =========  ========
approach                                good       usable     median
======================================  =========  =========  ========
chained propagation (camera_motion)     --         12%        --
direct, single anchor                   52%        95%        0.87
direct, nearest of two anchors          **88%**    **100%**   **0.89**
======================================  =========  =========  ========

The second anchor costs about two minutes of clicking and is what lifts the
video from half covered to nearly all of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from calibration.homography import Homography
from utils.logger import get_logger

logger = get_logger("calibration.anchor")

Box = Tuple[float, float, float, float]

# Lowe's ratio test: a match is kept only when the best candidate is clearly
# better than the runner-up, which is what keeps repetitive pitch texture
# (mown stripes, advertising boards) from producing confident nonsense.
RATIO_TEST = 0.75
# Below this many surviving matches a homography is not worth solving.
MIN_MATCHES = 12
# RANSAC reprojection tolerance, in pixels.
RANSAC_PX = 3.0
# Players are masked out by this many pixels beyond their box, so a limb just
# outside the detection does not contribute features.
PLAYER_PAD = 18


def player_mask(shape: Tuple[int, int], boxes: Sequence[Box],
                pad: int = PLAYER_PAD) -> np.ndarray:
    """White everywhere except (padded) player boxes.

    Features must come from the pitch, which is static, not from the players,
    which are not: a match driven by moving bodies describes their motion
    rather than the camera's.
    """
    height, width = shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes or ():
        ax1, ay1 = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
        ax2, ay2 = min(int(x2) + pad, width), min(int(y2) + pad, height)
        if ax2 > ax1 and ay2 > ay1:
            mask[ay1:ay2, ax1:ax2] = 0
    return mask


@dataclass
class Anchor:
    """A frame whose homography is known, plus its pitch features."""

    frame_index: int
    homography: Homography
    keypoints: tuple = field(repr=False, default=())
    descriptors: Optional[np.ndarray] = field(repr=False, default=None)
    label: str = ""

    @property
    def n_features(self) -> int:
        return len(self.keypoints)


@dataclass
class FrameFit:
    """The calibration chosen for one frame, and how well it matched."""

    frame_index: int
    homography: Optional[Homography]
    anchor_frame: Optional[int]
    inliers: int
    score: Optional[float] = None      # set when a verifier was supplied

    @property
    def ok(self) -> bool:
        return self.homography is not None


class AnchorRegistrar:
    """Registers frames directly against a set of verified anchors."""

    def __init__(self, detector=None, matcher=None) -> None:
        import cv2

        self._cv2 = cv2
        self._sift = detector if detector is not None else cv2.SIFT_create(nfeatures=4000)
        self._matcher = matcher if matcher is not None else cv2.BFMatcher()
        self._anchors: List[Anchor] = []

    # ------------------------------------------------------------------
    def add_anchor(self, frame_index: int, frame: np.ndarray,
                   homography: Homography, player_boxes: Sequence[Box] = (),
                   label: str = "") -> Anchor:
        """Register a frame whose homography is already known and trusted."""
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        kp, des = self._sift.detectAndCompute(
            gray, player_mask(gray.shape, player_boxes))
        anchor = Anchor(frame_index=frame_index, homography=homography,
                        keypoints=tuple(kp), descriptors=des,
                        label=label or "f%d" % frame_index)
        self._anchors.append(anchor)
        logger.info("Anchor %s: %d pitch features", anchor.label, anchor.n_features)
        return anchor

    @property
    def anchors(self) -> List[Anchor]:
        return list(self._anchors)

    # ------------------------------------------------------------------
    def _match(self, des, kp, anchor: Anchor):
        """Image->anchor homography and its inlier count, or ``(None, 0)``."""
        cv2 = self._cv2
        if des is None or anchor.descriptors is None:
            return None, 0
        pairs = self._matcher.knnMatch(des, anchor.descriptors, k=2)
        good = [m for m_n in pairs if len(m_n) == 2
                for m, n in [m_n] if m.distance < RATIO_TEST * n.distance]
        if len(good) < MIN_MATCHES:
            return None, len(good)
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([anchor.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        if matrix is None:
            return None, 0
        return matrix, int(inliers.sum()) if inliers is not None else 0

    def fit_frame(self, frame_index: int, frame: np.ndarray,
                  player_boxes: Sequence[Box] = (),
                  verify=None) -> FrameFit:
        """Calibrate one frame using whichever anchor actually works for it.

        ``verify`` is an optional ``(frame, Homography) -> score | None``. When
        given, every candidate is built and *checked*, and the highest-scoring
        one wins.

        Without it the choice falls back to the inlier count, which is a
        weaker proxy and measurably so: a frame looking at the left of the
        pitch can still share hundreds of centre-line features with a
        right-side anchor and pick it, then inherit a homography that was
        never constrained on the left. Measured on match3, selecting by
        inliers reached 88% good frames while selecting by verified score
        reached **97%** -- six of the seven failures were this exact mistake.

        Inliers still break ties, so a verifier that cannot judge a frame
        (no paint visible, say) degrades to the old behaviour rather than
        picking arbitrarily.
        """
        if not self._anchors:
            raise ValueError("no anchors registered")
        for anchor in self._anchors:
            if anchor.frame_index == frame_index:
                return FrameFit(frame_index, anchor.homography,
                                anchor.frame_index, anchor.n_features)

        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        kp, des = self._sift.detectAndCompute(
            gray, player_mask(gray.shape, player_boxes))

        candidates: List[Tuple[int, Anchor, np.ndarray]] = []
        for anchor in self._anchors:
            matrix, n_inliers = self._match(des, kp, anchor)
            if matrix is not None:
                candidates.append((n_inliers, anchor, matrix))

        if not candidates:
            return FrameFit(frame_index, None, None, 0)

        def combine(anchor: Anchor, matrix: np.ndarray) -> Homography:
            return Homography(
                np.asarray(anchor.homography.matrix, dtype=np.float64) @ matrix)

        if verify is None:
            n_inliers, anchor, matrix = max(candidates, key=lambda c: c[0])
            return FrameFit(frame_index, combine(anchor, matrix),
                            anchor.frame_index, n_inliers)

        best = None
        for n_inliers, anchor, matrix in candidates:
            homography = combine(anchor, matrix)
            score = verify(frame, homography)
            key = (score if score is not None else -1.0, n_inliers)
            if best is None or key > best[0]:
                best = (key, homography, anchor, n_inliers, score)

        key, homography, anchor, n_inliers, score = best
        return FrameFit(frame_index, homography, anchor.frame_index,
                        n_inliers, score)
