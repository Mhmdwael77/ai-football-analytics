"""Tests for the self-verifying automatic calibration flow."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.auto_calibrator import (
    CalibrationCoverage,
    VerifiedHomographyProvider,
    build_verified_provider,
    propagate_verified,
    select_anchors,
)
from calibration.calibration_quality import Segment

IDENTITY = np.eye(3)


def shift(dx: float, dy: float = 0.0) -> np.ndarray:
    """A translation, standing in for a homography in the pure tests."""
    m = np.eye(3)
    m[0, 2], m[1, 2] = dx, dy
    return m


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------
def test_select_anchors_keeps_only_verified_frames():
    scores = {1: 0.9, 2: 0.4, 3: None, 4: 0.75}
    assert select_anchors(scores, 0.7) == [1, 4]


def test_select_anchors_can_come_back_empty():
    assert select_anchors({1: 0.1, 2: None}, 0.7) == []


# ---------------------------------------------------------------------------
# Propagation stops where it stops being right
# ---------------------------------------------------------------------------
def test_propagation_spreads_an_anchor_across_its_segment():
    anchors = {5: IDENTITY.copy()}
    transforms = {f: IDENTITY.copy() for f in range(1, 11)}
    segments = [Segment(1, 10, anchor_frame=5, anchor_score=0.9)]
    out = propagate_verified(
        anchors, transforms, segments, verify=lambda f, m: 0.9)
    assert set(out) == set(range(1, 11))


def test_propagation_stops_when_the_score_drops():
    anchors = {5: IDENTITY.copy()}
    transforms = {f: IDENTITY.copy() for f in range(1, 21)}
    segments = [Segment(1, 20, anchor_frame=5, anchor_score=0.9)]
    # Good out to frame 8, then the calibration stops matching the paint.
    out = propagate_verified(
        anchors, transforms, segments,
        verify=lambda f, m: 0.9 if f <= 8 else 0.1)
    assert max(out) == 8
    assert 9 not in out


def test_propagation_never_crosses_a_segment_boundary():
    anchors = {5: IDENTITY.copy()}
    transforms = {f: IDENTITY.copy() for f in range(1, 21)}
    # A cut at 11: the anchor must not serve the second shot.
    segments = [Segment(1, 10, anchor_frame=5), Segment(11, 20)]
    out = propagate_verified(
        anchors, transforms, segments, verify=lambda f, m: 0.95)
    assert max(out) == 10


def test_propagation_carries_through_unjudgeable_frames():
    anchors = {5: IDENTITY.copy()}
    transforms = {f: IDENTITY.copy() for f in range(1, 16)}
    segments = [Segment(1, 15, anchor_frame=5)]
    # Frames 7-8 have no visible paint; the run should survive them.
    out = propagate_verified(
        anchors, transforms, segments,
        verify=lambda f, m: None if f in (7, 8) else 0.9)
    assert 9 in out and 10 in out
    assert 7 not in out and 8 not in out      # carried, but nothing claimed


def test_propagation_respects_max_reach():
    anchors = {1: IDENTITY.copy()}
    transforms = {f: IDENTITY.copy() for f in range(1, 100)}
    segments = [Segment(1, 99, anchor_frame=1)]
    out = propagate_verified(
        anchors, transforms, segments, verify=lambda f, m: 0.9, max_reach=5)
    assert max(out) == 6                       # anchor + 5 steps


def test_propagation_stops_at_a_missing_transform():
    anchors = {1: IDENTITY.copy()}
    transforms = {2: IDENTITY.copy(), 3: IDENTITY.copy()}   # nothing for 4
    segments = [Segment(1, 10, anchor_frame=1)]
    out = propagate_verified(
        anchors, transforms, segments, verify=lambda f, m: 0.9)
    assert max(out) == 3


def test_propagation_applies_the_camera_motion():
    anchors = {1: IDENTITY.copy()}
    transforms = {2: shift(4.0)}
    segments = [Segment(1, 2, anchor_frame=1)]
    out = propagate_verified(
        anchors, transforms, segments, verify=lambda f, m: 0.9)
    # Forward step is H @ inv(M): the translation is undone.
    assert out[2][0][0, 2] == pytest.approx(-4.0)


def test_better_scoring_anchor_wins_a_shared_frame():
    anchors = {1: shift(1.0), 9: shift(9.0)}
    transforms = {f: IDENTITY.copy() for f in range(1, 11)}
    segments = [Segment(1, 10)]
    out = propagate_verified(
        anchors, transforms, segments,
        verify=lambda f, m: 0.95 if m[0, 2] == 9.0 else 0.6)
    assert out[5][0][0, 2] == pytest.approx(9.0)


def test_propagation_survives_a_singular_transform():
    anchors = {1: IDENTITY.copy()}
    transforms = {2: np.zeros((3, 3))}
    segments = [Segment(1, 5, anchor_frame=1)]
    out = propagate_verified(
        anchors, transforms, segments, verify=lambda f, m: 0.9)
    assert set(out) == {1}


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------
def test_provider_knows_which_frames_it_cannot_serve():
    provider = VerifiedHomographyProvider({3: (IDENTITY.copy(), 0.9)})
    assert provider.is_valid(3)
    assert not provider.is_valid(4)
    # It still answers for_frame (nearest) for callers that need something.
    assert provider.for_frame(400) is provider.for_frame(3)


def test_provider_exposes_scores_not_a_reprojection_error():
    provider = VerifiedHomographyProvider(
        {1: (IDENTITY.copy(), 0.8), 2: (IDENTITY.copy(), 0.9)})
    assert provider.score(1) == pytest.approx(0.8)
    assert provider.score(99) is None
    assert provider.mean_score() == pytest.approx(0.85)
    assert np.isnan(provider.reprojection_error())
    assert len(provider) == 2


def test_provider_refuses_to_exist_with_nothing_verified():
    with pytest.raises(ValueError):
        VerifiedHomographyProvider({})


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def test_build_provider_covers_frames_and_asks_for_nothing():
    frames = list(range(1, 21))
    candidates = {f: IDENTITY.copy() for f in (5, 15)}
    scores = {f: (0.9 if f in (5, 15) else 0.2) for f in frames}
    transforms = {f: IDENTITY.copy() for f in frames}

    provider, coverage = build_verified_provider(
        candidates, scores, transforms, frames, verify=lambda f, m: 0.9)

    assert provider is not None
    assert coverage.coverage == pytest.approx(1.0)
    assert coverage.review_frames == []
    assert coverage.summary()["need_manual_keyframe"] == 0


def test_build_provider_flags_only_the_shot_it_could_not_calibrate():
    frames = list(range(1, 41))
    candidates = {5: IDENTITY.copy()}
    scores = {f: (0.9 if f == 5 else 0.1) for f in frames}
    transforms = {f: IDENTITY.copy() for f in frames}

    provider, coverage = build_verified_provider(
        candidates, scores, transforms, frames,
        # The second shot never verifies, whatever we propagate into it.
        verify=lambda f, m: 0.9 if f <= 20 else 0.05,
        cut_frames=[21])

    assert provider is not None
    assert coverage.review_frames == [21]      # one keyframe, for one segment
    assert all(provider.is_valid(f) for f in range(1, 21))
    assert not provider.is_valid(30)


def test_build_provider_returns_nothing_when_no_frame_verifies():
    frames = list(range(1, 11))
    scores = {f: 0.1 for f in frames}
    provider, coverage = build_verified_provider(
        {}, scores, {}, frames, verify=lambda f, m: 0.1)

    assert provider is None
    assert coverage.anchors == 0
    # Nothing was covered, so the whole video is offered for review.
    assert coverage.coverage == 0.0
    assert coverage.review_frames == [1]


def test_coverage_summary_is_serialisable():
    coverage = CalibrationCoverage(
        frames_total=100, frames_covered=75, anchors=3,
        segments=[Segment(1, 50, anchor_frame=10), Segment(51, 100)])
    summary = coverage.summary()
    assert summary["coverage"] == pytest.approx(0.75)
    assert summary["need_manual_keyframe"] == 1
    assert summary["review_frames"] == [51]
