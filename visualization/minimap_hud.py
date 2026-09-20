"""A broadcast-style minimap HUD: compact, translucent, anti-aliased.

The first overlay this replaced was an opaque block in the middle of the
frame -- it hid the very play it was describing. This one is sized like a
game HUD, alpha-composited so the pitch reads through it, and drawn with
anti-aliased hairlines so it looks printed rather than pasted on.

Three things were learned the hard way and are worth keeping:

* **Dot size is a distance, not a preference.** At a 420 px panel the pitch
  is ~3.8 px/m, so a 5 px radius draws a player 3.1 m wide -- six times life
  size, and neighbours merge into one blob. 3 px is ~1.6 m and they separate.
* **A white ring around each dot inflates it.** It added 2 px to every
  marker, so two players standing near each other touched. Team colour
  already separates them.
* **Blend the panel, not the markers.** See :meth:`MinimapHUD.draw_dots`:
  a cream shirt composited at 62% over green grass comes out the same brown
  as a dark red one, which made real kit colours unreadable.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

Colour = Tuple[int, int, int]

# --- palette (BGR) --------------------------------------------------------
PANEL_BG = (28, 46, 22)        # deep pitch green
PANEL_EDGE = (210, 210, 210)
LINE = (235, 235, 235)
SHADOW = (12, 12, 12)


def _rounded_mask(h: int, w: int, radius: int) -> np.ndarray:
    """Single-channel mask with rounded corners, anti-aliased."""
    ss = 4                                   # supersample for smooth corners
    m = np.zeros((h * ss, w * ss), np.uint8)
    r = radius * ss
    cv2.rectangle(m, (r, 0), (w * ss - r, h * ss), 255, -1)
    cv2.rectangle(m, (0, r), (w * ss, h * ss - r), 255, -1)
    for cx, cy in ((r, r), (w * ss - r, r), (r, h * ss - r), (w * ss - r, h * ss - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)


class MinimapHUD:
    """Draws a translucent top-down pitch and blends it into a frame."""

    def __init__(self, length: float = 105.0, width: float = 68.0,
                 panel_width: int = 360, opacity: float = 0.62,
                 margin: int = 10, radius: int = 10) -> None:
        self.length, self.width = float(length), float(width)
        self.opacity = float(opacity)
        self.margin = int(margin)
        self.pad = 12
        # keep the pitch aspect ratio, add padding for the touchlines
        inner_w = panel_width - 2 * self.pad
        inner_h = int(round(inner_w * self.width / self.length))
        self.w = panel_width
        self.h = inner_h + 2 * self.pad
        self.scale = inner_w / self.length
        self.radius = int(radius)
        self._mask = _rounded_mask(self.h, self.w, self.radius)
        self._panel = self._draw_pitch()

    # ------------------------------------------------------------------
    def field_to_px(self, x: float, y: float) -> Tuple[int, int]:
        return (int(round(self.pad + x * self.scale)),
                int(round(self.pad + y * self.scale)))

    def on_pitch(self, x: float, y: float) -> bool:
        return -1.0 <= x <= self.length + 1.0 and -1.0 <= y <= self.width + 1.0

    # ------------------------------------------------------------------
    def _draw_pitch(self) -> np.ndarray:
        p = np.full((self.h, self.w, 3), PANEL_BG, np.uint8)
        # faint mown stripes so the panel reads as grass, not a chart
        stripe = np.zeros_like(p)
        n = 10
        for i in range(n):
            if i % 2:
                continue
            x0 = self.pad + int(i * self.length / n * self.scale)
            x1 = self.pad + int((i + 1) * self.length / n * self.scale)
            cv2.rectangle(stripe, (x0, 0), (x1, self.h), (10, 14, 8), -1)
        p = cv2.add(p, stripe)

        L, W, s = self.length, self.width, self.scale
        f = self.field_to_px
        line = LINE
        t = 1
        aa = cv2.LINE_AA
        cv2.rectangle(p, f(0, 0), f(L, W), line, t, aa)
        cv2.line(p, f(L / 2, 0), f(L / 2, W), line, t, aa)
        cv2.circle(p, f(L / 2, W / 2), int(9.15 * s), line, t, aa)
        cv2.circle(p, f(L / 2, W / 2), 2, line, -1, aa)
        for x0, x1 in ((0, 16.5), (L, L - 16.5)):
            cv2.rectangle(p, f(x0, W / 2 - 20.16), f(x1, W / 2 + 20.16), line, t, aa)
        for x0, x1 in ((0, 5.5), (L, L - 5.5)):
            cv2.rectangle(p, f(x0, W / 2 - 9.16), f(x1, W / 2 + 9.16), line, t, aa)
        for x in (11.0, L - 11.0):
            cv2.circle(p, f(x, W / 2), 2, line, -1, aa)
        return p

    # ------------------------------------------------------------------
    def render(self, objects: Iterable[Tuple[float, float, Colour, str]],
               ball_radius: int = 2, player_radius: int = 3,
               ring: bool = False) -> np.ndarray:
        """objects: (x_m, y_m, colour, role).

        The default radius keeps a dot near a player's real footprint: at
        this panel size 3 px is about 1.6 m across, where 5 px was 3.1 m --
        six times a player's width, which is why neighbours merged into one
        blob. The white ring is off for the same reason: it added 2 px to
        every dot, so two players standing near each other touched. Team
        colour already separates them; a ring only inflates them.
        """
        panel = self._panel.copy()
        for x, y, colour, role in objects:
            if not self.on_pitch(x, y):
                continue
            px, py = self.field_to_px(x, y)
            r = ball_radius if role == "ball" else player_radius
            cv2.circle(panel, (px + 1, py + 1), r, SHADOW, -1, cv2.LINE_AA)
            cv2.circle(panel, (px, py), r, colour, -1, cv2.LINE_AA)
            if ring:
                cv2.circle(panel, (px, py), r, (245, 245, 245), 1, cv2.LINE_AA)
        return panel

    # ------------------------------------------------------------------
    def draw_dots(self, frame: np.ndarray,
                  objects: Iterable[Tuple[float, float, Colour, str]],
                  position: str = "bottom_center",
                  ball_radius: int = 2, player_radius: int = 3) -> np.ndarray:
        """Paint markers at full strength onto an already-blended frame.

        Use this instead of passing the objects to :meth:`render` whenever the
        marker colour carries meaning -- a real kit colour, say. Markers drawn
        into the panel get alpha-blended along with it, and blending washes
        colours toward the grass underneath until a white shirt and a dark red
        one are the same brown. Painted on top they keep their true colour
        while the panel stays see-through.
        """
        x0, y0 = self.origin(frame)
        for x, y, colour, role in objects:
            if not self.on_pitch(x, y):
                continue
            px, py = self.field_to_px(x, y)
            r = ball_radius if role == "ball" else player_radius
            cv2.circle(frame, (x0 + px + 1, y0 + py + 1), r, SHADOW, -1, cv2.LINE_AA)
            cv2.circle(frame, (x0 + px, y0 + py), r, colour, -1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    def origin(self, frame: np.ndarray,
               position: str = "bottom_center") -> Tuple[int, int]:
        """Top-left pixel of the panel inside ``frame``."""
        H, W = frame.shape[:2]
        if position == "bottom_right":
            return W - self.w - self.margin, H - self.h - self.margin
        if position == "bottom_left":
            return self.margin, H - self.h - self.margin
        return (W - self.w) // 2, H - self.h - self.margin

    # ------------------------------------------------------------------
    def blend_into(self, frame: np.ndarray, panel: np.ndarray,
                   position: str = "bottom_center") -> np.ndarray:
        """Alpha-composite the panel onto the frame with rounded corners."""
        x0, y0 = self.origin(frame, position)

        roi = frame[y0:y0 + self.h, x0:x0 + self.w]
        alpha = (self._mask.astype(np.float32) / 255.0) * self.opacity
        alpha3 = alpha[:, :, None]
        blended = (panel.astype(np.float32) * alpha3
                   + roi.astype(np.float32) * (1.0 - alpha3))
        frame[y0:y0 + self.h, x0:x0 + self.w] = blended.astype(np.uint8)

        # a hairline edge so the panel has a defined boundary
        edge = np.zeros((self.h, self.w), np.uint8)
        cv2.rectangle(edge, (0, 0), (self.w - 1, self.h - 1), 255, 1)
        edge = cv2.bitwise_and(edge, self._mask)
        ys, xs = np.nonzero(edge)
        if len(ys):
            roi2 = frame[y0:y0 + self.h, x0:x0 + self.w]
            roi2[ys, xs] = (roi2[ys, xs] * 0.45
                            + np.array(PANEL_EDGE, np.float32) * 0.55).astype(np.uint8)
        return frame
