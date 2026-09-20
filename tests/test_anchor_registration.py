"""Tests for direct anchor registration.

The decision logic (which anchor a frame should use, and what happens when
none of them match) is exercised with synthetic frames, so no video, no model
weights and no GPU are involved.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.anchor_registration import (
    Anchor,
    AnchorRegistrar,
    FrameFit,
    player_mask,
)
from calibration.homography import Homography


# ---------------------------------------------------------------------------
# player_mask
# ---------------------------------------------------------------------------
def test_player_mask_is_white_where_there_are_no_players():
    mask = player_mask((100, 200), [])
    assert mask.shape == (100, 200)
    assert (mask == 255).all()


def test_player_mask_blanks_the_player_box_with_padding():
    mask = player_mask((100, 200), [(50, 40, 70, 60)], pad=5)
    assert mask[50, 60] == 0, "inside the box must be masked out"
    assert mask[36, 46] == 0, "the pad must be masked out too"
    assert mask[10, 10] == 255, "far from any player must stay usable"


def test_player_mask_clips_boxes_at_the_frame_edge():
    # A box hanging off the top-left must not wrap around or raise.
    mask = player_mask((50, 50), [(-20, -20, 10, 10)], pad=4)
    assert mask[0, 0] == 0
    assert mask[49, 49] == 255


# ---------------------------------------------------------------------------
# helpers: synthetic frames with enough texture for SIFT
# ---------------------------------------------------------------------------
def _textured(seed: int, size=(480, 640)) -> np.ndarray:
    """A repeatable noisy image -- SIFT needs real corners to find."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[0] // 8, size[1] // 8, 3), dtype=np.uint8)
    return cv2.resize(base, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def _shifted(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))


def _identity_homography() -> Homography:
    return Homography(np.eye(3))


# ---------------------------------------------------------------------------
# registrar behaviour
# ---------------------------------------------------------------------------
def test_fitting_without_any_anchor_is_an_error():
    reg = AnchorRegistrar()
    with pytest.raises(ValueError):
        reg.fit_frame(1, _textured(0))


def test_the_anchor_frame_itself_returns_its_own_homography():
    reg = AnchorRegistrar()
    img = _textured(1)
    H = _identity_homography()
    reg.add_anchor(7, img, H, [], "only")

    fit = reg.fit_frame(7, img)

    assert fit.ok
    assert fit.anchor_frame == 7
    assert fit.homography is H, "no registration is needed against itself"


def test_a_frame_that_matches_nothing_reports_failure_instead_of_guessing():
    """Two unrelated images must not produce a confident homography."""
    reg = AnchorRegistrar()
    reg.add_anchor(1, _textured(10), _identity_homography(), [], "a")

    fit = reg.fit_frame(99, _textured(999))

    # Either no homography at all, or at most a handful of inliers -- what it
    # must never do is return a well-supported fit for unrelated content.
    assert (not fit.ok) or fit.inliers < 40


def test_a_shifted_frame_registers_back_onto_its_anchor():
    img = _textured(3)
    reg = AnchorRegistrar()
    reg.add_anchor(1, img, _identity_homography(), [], "a")

    fit = reg.fit_frame(2, _shifted(img, 12, -8))

    assert fit.ok, "a translated copy must register"
    assert fit.anchor_frame == 1
    assert fit.inliers > 20
    # The recovered transform should undo roughly that translation.
    matrix = np.asarray(fit.homography.matrix, float)
    assert matrix[0, 2] == pytest.approx(-12, abs=3)
    assert matrix[1, 2] == pytest.approx(8, abs=3)


def test_the_closest_anchor_wins_when_several_are_registered():
    """The anchor sharing the most features is the one whose view matches.

    This is the property the whole module rests on: a frame is calibrated by
    the anchor that actually saw the same part of the pitch, never by one that
    would have to extrapolate into pitch it never covered.
    """
    near = _textured(5)
    far = _textured(6)
    reg = AnchorRegistrar()
    reg.add_anchor(10, far, Homography(np.eye(3)), [], "far")
    reg.add_anchor(20, near, Homography(np.diag([2.0, 2.0, 1.0])), [], "near")

    fit = reg.fit_frame(21, _shifted(near, 6, 6))

    assert fit.ok
    assert fit.anchor_frame == 20, "must pick the anchor it shares content with"


def test_anchors_are_listed_in_registration_order():
    reg = AnchorRegistrar()
    reg.add_anchor(4, _textured(7), _identity_homography(), [], "first")
    reg.add_anchor(9, _textured(8), _identity_homography(), [], "second")

    assert [a.frame_index for a in reg.anchors] == [4, 9]
    assert [a.label for a in reg.anchors] == ["first", "second"]
    assert all(a.n_features > 0 for a in reg.anchors)


def test_frame_fit_reports_not_ok_without_a_homography():
    assert FrameFit(1, None, None, 0).ok is False
    assert FrameFit(1, _identity_homography(), 2, 50).ok is True


def test_masked_players_do_not_contribute_features():
    """Covering most of the frame must leave far fewer features behind."""
    img = _textured(11)
    reg = AnchorRegistrar()
    full = reg.add_anchor(1, img, _identity_homography(), [], "full")
    covered = reg.add_anchor(
        2, img, _identity_homography(),
        [(0, 0, img.shape[1], int(img.shape[0] * 0.8))], "covered")

    assert covered.n_features < full.n_features


# ---------------------------------------------------------------------------
# choosing the anchor by a verified score rather than by inlier count
# ---------------------------------------------------------------------------
def test_a_verifier_overrides_the_inlier_count():
    """The anchor that VERIFIES best wins, even with fewer inliers.

    Selecting on inliers alone is what let a left-looking frame adopt a
    right-side anchor: the two still share plenty of centre features, so the
    match looks strong while the resulting homography is unconstrained on the
    half the frame actually shows.
    """
    img = _textured(21)
    reg = AnchorRegistrar()
    # Both anchors see the same content, so both will match well.
    reg.add_anchor(1, img, Homography(np.eye(3)), [], "plain")
    reg.add_anchor(2, img, Homography(np.diag([3.0, 3.0, 1.0])), [], "scaled")

    # A verifier that only likes the SECOND anchor's homography.
    def verify(frame, homography):
        return 0.95 if np.asarray(homography.matrix)[0, 0] > 2.0 else 0.10

    fit = reg.fit_frame(3, _shifted(img, 5, 5), verify=verify)

    assert fit.ok
    assert fit.anchor_frame == 2, "the verified-best anchor must win"
    assert fit.score == pytest.approx(0.95)


def test_the_fit_falls_back_to_inliers_when_the_verifier_cannot_judge():
    """A verifier returning None must not make the choice arbitrary."""
    img = _textured(22)
    reg = AnchorRegistrar()
    reg.add_anchor(1, img, Homography(np.eye(3)), [], "a")

    fit = reg.fit_frame(2, _shifted(img, 4, 4), verify=lambda f, h: None)

    assert fit.ok, "an unjudgeable frame still gets its best-matching anchor"
    assert fit.score is None


def test_score_is_absent_when_no_verifier_is_given():
    img = _textured(23)
    reg = AnchorRegistrar()
    reg.add_anchor(1, img, Homography(np.eye(3)), [], "a")

    fit = reg.fit_frame(2, _shifted(img, 3, 3))

    assert fit.ok
    assert fit.score is None
