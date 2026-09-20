"""Tests for scoring a homography against the painted lines."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.calibration_quality import (
    MIN_SAMPLES,
    SCORE_GOOD,
    SCORE_USABLE,
    Segment,
    classify_score,
    densify,
    line_alignment_score,
    pitch_polylines,
    project_field_to_image,
    review_effort,
    split_segments,
    suggest_review_segments,
    white_line_mask,
)
from calibration.homography import Homography
from calibration.pitch_model import PitchDimensions


# ---------------------------------------------------------------------------
# Pitch template
# ---------------------------------------------------------------------------
def test_template_stays_inside_the_pitch():
    dims = PitchDimensions()
    for poly in pitch_polylines():
        assert poly[:, 0].min() >= -0.01
        assert poly[:, 0].max() <= dims.length + 0.01
        assert poly[:, 1].min() >= -0.01
        assert poly[:, 1].max() <= dims.width + 0.01


def test_template_includes_the_centre_circle():
    dims = PitchDimensions()
    centre = np.array([dims.length / 2.0, dims.width / 2.0])
    for poly in pitch_polylines():
        radii = np.linalg.norm(poly - centre, axis=1)
        if np.allclose(radii, dims.center_circle_radius, atol=1e-6):
            return
    pytest.fail("no polyline traces the centre circle")


def test_densify_samples_along_a_long_line():
    dense = densify(np.array([[0.0, 0.0], [105.0, 0.0]]), per_metre=2.0)
    assert len(dense) >= 200
    assert dense[0] == pytest.approx((0.0, 0.0))
    assert dense[-1] == pytest.approx((105.0, 0.0))


def test_densify_passes_through_a_degenerate_polyline():
    single = np.array([[1.0, 2.0]])
    assert densify(single).shape == (1, 2)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def _identity_like_homography(scale=10.0):
    """Field metres -> pixels at ``scale`` px/m (a plain top-down view)."""
    matrix = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1.0]])
    return Homography(matrix)


def test_project_field_to_image_round_trips():
    homography = _identity_like_homography(10.0)
    pts = np.array([[0.0, 0.0], [52.5, 34.0], [105.0, 68.0]])
    xy, valid = project_field_to_image(homography.inverse, pts)
    assert valid.all()
    assert xy[1] == pytest.approx((525.0, 340.0))


def test_project_field_to_image_flags_points_behind_the_camera():
    # A homography whose third row makes w change sign across the pitch.
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.02, 0.0, -1.0]])
    xy, valid = project_field_to_image(
        np.linalg.inv(matrix), np.array([[0.0, 0.0], [100.0, 10.0]]))
    assert not valid.all()          # the two points straddle the horizon


# ---------------------------------------------------------------------------
# Paint mask
# ---------------------------------------------------------------------------
def _synthetic_pitch(width=640, height=360, scale=5.0):
    """A green frame with white lines drawn exactly where the template says."""
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:] = (60, 130, 60)                       # BGR grass
    for poly in pitch_polylines():
        pts = np.round(poly * scale).astype(np.int32)
        cv2.polylines(frame, [pts], False, (245, 245, 245), 2)
    return frame


def test_white_line_mask_finds_paint_and_ignores_grass():
    frame = _synthetic_pitch()
    mask = white_line_mask(frame)
    assert mask.sum() > 500                        # the lines were found
    assert mask.mean() < 0.2                       # but most of it is grass


def test_white_line_mask_is_binary_and_frame_shaped():
    frame = _synthetic_pitch()
    mask = white_line_mask(frame)
    assert mask.shape == frame.shape[:2]
    assert set(np.unique(mask)).issubset({0, 1})


def test_white_line_mask_rejects_a_non_image():
    with pytest.raises(ValueError):
        white_line_mask(np.zeros((10, 10), np.uint8))


# ---------------------------------------------------------------------------
# The score itself
# ---------------------------------------------------------------------------
def test_correct_homography_scores_far_above_a_wrong_one():
    scale = 5.0
    frame = _synthetic_pitch(scale=scale)
    correct = _identity_like_homography(scale)
    # Same view shifted 25 m down the pitch: lines land on bare grass.
    shifted = Homography(
        np.array([[1 / scale, 0, -25.0], [0, 1 / scale, 0], [0, 0, 1.0]]))

    good, n_good = line_alignment_score(frame, correct)
    bad, n_bad = line_alignment_score(frame, shifted)

    assert n_good >= MIN_SAMPLES and n_bad >= MIN_SAMPLES
    assert good is not None and bad is not None
    assert good > SCORE_GOOD
    assert bad < good - 0.3


def test_a_collapsed_homography_is_rejected_not_flattered():
    """A near-singular matrix crams the template onto a pencil of lines.

    Those samples can land on real paint and score near 1.0, which is how a
    geometrically meaningless matrix used to pass the gate and become an
    anchor. It must come back as "unknown" instead.
    """
    frame = _synthetic_pitch(scale=5.0)
    # Third row nearly parallel to the first two -> the map collapses.
    collapsed = Homography(
        np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [1.0, 1.0, 1e-6]]))

    score, samples = line_alignment_score(frame, collapsed)

    assert score is None, "a collapsed homography must not receive a score"


def test_spread_guard_keeps_a_genuine_fit():
    """The guard must not reject an honest, well-spread homography."""
    scale = 5.0
    frame = _synthetic_pitch(scale=scale)
    score, samples = line_alignment_score(frame, _identity_like_homography(scale))

    assert score is not None
    assert score > SCORE_GOOD


def test_score_is_none_when_there_is_no_paint_to_judge_against():
    plain = np.zeros((360, 640, 3), np.uint8)
    plain[:] = (60, 130, 60)
    score, samples = line_alignment_score(plain, _identity_like_homography())
    assert score is None
    assert samples == 0


def test_score_is_none_when_the_template_misses_the_image():
    frame = _synthetic_pitch()
    # Push the pitch thousands of pixels away: nothing lands in view.
    far = Homography(np.array([[0.2, 0, -1e4], [0, 0.2, -1e4], [0, 0, 1.0]]))
    score, _samples = line_alignment_score(frame, far)
    assert score is None


def test_score_accepts_a_precomputed_mask():
    frame = _synthetic_pitch()
    mask = white_line_mask(frame)
    a, _ = line_alignment_score(frame, _identity_like_homography(5.0))
    b, _ = line_alignment_score(frame, _identity_like_homography(5.0), mask=mask)
    assert a == pytest.approx(b)


def test_classify_score_bands():
    assert classify_score(None) == "unknown"
    assert classify_score(SCORE_GOOD + 0.05) == "good"
    assert classify_score(SCORE_USABLE + 0.01) == "usable"
    assert classify_score(0.1) == "wrong"


# ---------------------------------------------------------------------------
# Segments -- one keyframe per shot, not per frame
# ---------------------------------------------------------------------------
def test_split_segments_without_cuts_is_one_run():
    assert split_segments([1, 5, 9]) == [(1, 9)]


def test_split_segments_breaks_at_each_cut():
    assert split_segments(range(1, 101), [40, 70]) == [(1, 39), (40, 69), (70, 100)]


def test_split_segments_ignores_cuts_outside_the_range():
    assert split_segments([10, 20, 30], [5, 500]) == [(10, 30)]


def test_split_segments_of_nothing():
    assert split_segments([]) == []


def test_segment_with_a_good_frame_is_anchored_automatically():
    scores = {1: 0.2, 10: 0.9, 20: 0.3}
    segments = suggest_review_segments(scores)
    assert len(segments) == 1
    assert segments[0].anchor_frame == 10
    assert not segments[0].needs_review


def test_segment_picks_its_best_frame_as_the_anchor():
    scores = {1: 0.75, 10: 0.93, 20: 0.80}
    assert suggest_review_segments(scores)[0].anchor_frame == 10


def test_segment_with_no_good_frame_is_flagged_for_one_keyframe():
    scores = {1: 0.2, 10: 0.3, 20: None}
    segment = suggest_review_segments(scores)[0]
    assert segment.needs_review
    assert segment.anchor_frame is None


def test_only_the_failing_segment_asks_for_a_human():
    # Two shots: the first calibrates itself, the second never does.
    scores = {1: 0.9, 20: 0.8, 60: 0.1, 80: 0.2}
    segments = suggest_review_segments(scores, cut_frames=[50])
    assert len(segments) == 2
    assert not segments[0].needs_review
    assert segments[1].needs_review

    effort = review_effort(segments)
    assert effort["segments"] == 2
    assert effort["need_manual_keyframe"] == 1
    assert effort["review_frames"] == [50]
    assert 0.0 < effort["auto_coverage"] < 1.0


def test_review_effort_on_a_fully_automatic_video():
    segments = [Segment(1, 100, anchor_frame=10, anchor_score=0.9)]
    effort = review_effort(segments)
    assert effort["need_manual_keyframe"] == 0
    assert effort["auto_coverage"] == pytest.approx(1.0)
    assert effort["review_frames"] == []


def test_segment_geometry():
    segment = Segment(10, 19)
    assert segment.length == 10
    assert segment.contains(10) and segment.contains(19)
    assert not segment.contains(9) and not segment.contains(20)


def test_suggest_review_segments_of_nothing():
    assert suggest_review_segments({}) == []
