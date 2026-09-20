"""Tests for the translucent minimap HUD.

Geometry, placement and compositing are checked on synthetic frames -- no
video, no model, no display.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from visualization.minimap_hud import MinimapHUD


def _frame(h=720, w=1280, colour=(30, 120, 30)):
    return np.full((h, w, 3), colour, np.uint8)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_the_panel_keeps_the_pitch_aspect_ratio():
    hud = MinimapHUD(length=105.0, width=68.0, panel_width=420)
    inner_w = hud.w - 2 * hud.pad
    inner_h = hud.h - 2 * hud.pad
    assert inner_h / inner_w == pytest.approx(68.0 / 105.0, abs=0.01)


def test_the_pitch_corners_map_to_the_padded_box():
    hud = MinimapHUD(panel_width=420)
    assert hud.field_to_px(0.0, 0.0) == (hud.pad, hud.pad)
    x, y = hud.field_to_px(hud.length, hud.width)
    assert x == pytest.approx(hud.w - hud.pad, abs=2)
    assert y == pytest.approx(hud.h - hud.pad, abs=2)


def test_the_centre_spot_lands_in_the_middle():
    hud = MinimapHUD(panel_width=420)
    x, y = hud.field_to_px(hud.length / 2, hud.width / 2)
    assert x == pytest.approx(hud.w / 2, abs=2)
    assert y == pytest.approx(hud.h / 2, abs=2)


def test_on_pitch_accepts_the_touchline_and_rejects_the_stands():
    hud = MinimapHUD()
    assert hud.on_pitch(0.0, 0.0)
    assert hud.on_pitch(105.0, 68.0)
    assert not hud.on_pitch(-5.0, 34.0)
    assert not hud.on_pitch(52.5, 90.0)


def test_a_wider_panel_gives_a_finer_scale():
    assert MinimapHUD(panel_width=600).scale > MinimapHUD(panel_width=300).scale


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------
def test_the_panel_sits_where_it_is_asked_to():
    hud = MinimapHUD(panel_width=420, margin=10)
    f = _frame()
    H, W = f.shape[:2]

    cx, cy = hud.origin(f, "bottom_center")
    rx, _ = hud.origin(f, "bottom_right")
    lx, _ = hud.origin(f, "bottom_left")

    assert cx == (W - hud.w) // 2
    assert rx == W - hud.w - hud.margin
    assert lx == hud.margin
    assert cy == H - hud.h - hud.margin


def test_the_panel_never_hangs_off_the_frame():
    hud = MinimapHUD(panel_width=420, margin=10)
    f = _frame()
    for pos in ("bottom_center", "bottom_right", "bottom_left"):
        x0, y0 = hud.origin(f, pos)
        assert x0 >= 0 and y0 >= 0
        assert x0 + hud.w <= f.shape[1]
        assert y0 + hud.h <= f.shape[0]


# ---------------------------------------------------------------------------
# compositing -- the point of the whole thing
# ---------------------------------------------------------------------------
def test_the_panel_is_translucent_not_opaque():
    """The pitch must still be readable through the HUD."""
    hud = MinimapHUD(panel_width=420, opacity=0.62)
    original = _frame(colour=(40, 200, 40))
    out = hud.blend_into(original.copy(), hud.render([]))

    x0, y0 = hud.origin(out)
    centre = out[y0 + hud.h // 2, x0 + hud.w // 2]
    panel_only = hud.render([])[hud.h // 2, hud.w // 2]

    assert not np.array_equal(centre, panel_only), "fully opaque"
    assert not np.array_equal(centre, original[0, 0]), "panel invisible"


def test_a_higher_opacity_hides_more_of_the_pitch():
    grass = _frame(colour=(40, 200, 40))
    faint = MinimapHUD(panel_width=420, opacity=0.2)
    solid = MinimapHUD(panel_width=420, opacity=0.9)

    a = faint.blend_into(grass.copy(), faint.render([]))
    b = solid.blend_into(grass.copy(), solid.render([]))
    x0, y0 = faint.origin(a)
    pa = a[y0 + faint.h // 2, x0 + faint.w // 2].astype(int)
    pb = b[y0 + solid.h // 2, x0 + solid.w // 2].astype(int)

    g = int(grass[0, 0][1])
    assert abs(pa[1] - g) < abs(pb[1] - g)


def test_nothing_outside_the_panel_is_touched():
    hud = MinimapHUD(panel_width=420, margin=10)
    before = _frame()
    after = hud.blend_into(before.copy(), hud.render([]))
    assert np.array_equal(after[:100, :100], before[:100, :100])


# ---------------------------------------------------------------------------
# markers
# ---------------------------------------------------------------------------
def test_a_dot_appears_where_the_player_is():
    hud = MinimapHUD(panel_width=420)
    panel = hud.render([(52.5, 34.0, (0, 0, 255), "player")])
    px, py = hud.field_to_px(52.5, 34.0)
    assert panel[py, px][2] > 200, "a red dot should be red at its centre"


def test_a_player_off_the_pitch_is_not_drawn():
    hud = MinimapHUD(panel_width=420)
    empty = hud.render([])
    with_ghost = hud.render([(-40.0, 34.0, (0, 0, 255), "player")])
    assert np.array_equal(empty, with_ghost)


def test_draw_dots_keeps_the_marker_colour_exact():
    """Opaque markers are what make a real kit colour survive.

    Drawn into the panel they would be alpha-blended toward the grass, which
    is what turned a cream shirt and a dark red one into the same brown.
    """
    hud = MinimapHUD(panel_width=420, opacity=0.62)
    out = hud.blend_into(_frame(), hud.render([]))
    colour = (37, 211, 102)
    out = hud.draw_dots(out, [(52.5, 34.0, colour, "player")])

    x0, y0 = hud.origin(out)
    px, py = hud.field_to_px(52.5, 34.0)
    assert tuple(int(v) for v in out[y0 + py, x0 + px]) == colour


def test_the_ball_is_drawn_smaller_than_a_player():
    hud = MinimapHUD(panel_width=420)
    ball = hud.render([(52.5, 34.0, (255, 255, 255), "ball")])
    player = hud.render([(52.5, 34.0, (255, 255, 255), "player")])
    assert (ball > 200).sum() < (player > 200).sum()


def test_the_ring_is_off_by_default():
    """The ring inflated every dot by 2 px and merged close players."""
    hud = MinimapHUD(panel_width=420)
    plain = hud.render([(52.5, 34.0, (0, 0, 255), "player")])
    ringed = hud.render([(52.5, 34.0, (0, 0, 255), "player")], ring=True)
    assert not np.array_equal(plain, ringed)
    assert (ringed > 230).sum() > (plain > 230).sum()
