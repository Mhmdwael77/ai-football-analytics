"""Physically-constrained smoothing of player positions on the pitch.

What this fixes
---------------
A detection box is not a rigid marker on a player: it is the tightest
rectangle around whatever pixels of them are visible, and those pixels change
every frame. Measured on match3, a player who moved 8 px in a whole second had
his box width swing from 13.4 px to 22.0 px -- his arms came out, nothing
else. The box centre slides with that, and the pipeline reads the slide as
movement.

At that clip's scale one pixel is about four centimetres, so a one-pixel
wobble becomes ~1 m/s of invented speed. Differentiating a second time to get
acceleration squares the problem: **81% of measured accelerations were above
12 m/s², which no human can produce.**

Why here and not on the box
---------------------------
The limits we can rely on are physical, and physics is in metres. The same
two-pixel wobble is ten centimetres for a near player and a full metre for a
far one, so a pixel-space filter would under-smooth exactly where the error is
worst. Field coordinates make one threshold correct everywhere.

Why not just smooth harder
--------------------------
Exponential smoothing blurs real football: players change direction sharply
and that is the signal analysts want. This filter leaves plausible motion
alone and only intervenes where the required acceleration is impossible, so a
genuine sprint survives while a jittering box does not.

A gap in the track (occlusion, a lost frame, a calibration the verifier could
not trust) breaks the chain: the filter restarts rather than smoothing across
the hole and inventing a path through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

# A footballer accelerates at roughly 5-10 m/s^2. Twelve is already generous:
# above it the measurement is the box moving, not the player.
MAX_ACCEL_MS2 = 12.0
# Top human sprint is ~10 m/s; footballers rarely pass 9.
MAX_SPEED_MS = 10.0
# Frames further apart than this are separate journeys, not one motion.
MAX_GAP_FRAMES = 5


@dataclass
class FilterStats:
    """What the filter actually had to do -- useful as a health signal."""

    points: int = 0
    accel_clamped: int = 0
    speed_clamped: int = 0
    segments: int = 0

    @property
    def clamped_fraction(self) -> float:
        if not self.points:
            return 0.0
        return (self.accel_clamped + self.speed_clamped) / self.points


def _limit(prev: Point, prev_v: Optional[Point], measured: Point, dt: float,
           max_accel: float, max_speed: float, stats: FilterStats) -> Tuple[Point, Point]:
    """Move toward ``measured`` as far as physics allows from ``prev``."""
    vx = (measured[0] - prev[0]) / dt
    vy = (measured[1] - prev[1]) / dt

    if prev_v is not None:
        dvx, dvy = vx - prev_v[0], vy - prev_v[1]
        accel = (dvx * dvx + dvy * dvy) ** 0.5 / dt
        if accel > max_accel:
            scale = max_accel * dt / accel
            vx, vy = prev_v[0] + dvx * scale, prev_v[1] + dvy * scale
            stats.accel_clamped += 1

    speed = (vx * vx + vy * vy) ** 0.5
    if speed > max_speed:
        vx, vy = vx * max_speed / speed, vy * max_speed / speed
        stats.speed_clamped += 1

    return (prev[0] + vx * dt, prev[1] + vy * dt), (vx, vy)


def filter_track(
    samples: Sequence[Tuple[int, Point]],
    fps: float = 25.0,
    max_accel: float = MAX_ACCEL_MS2,
    max_speed: float = MAX_SPEED_MS,
    max_gap: int = MAX_GAP_FRAMES,
    stats: Optional[FilterStats] = None,
) -> List[Tuple[int, Point]]:
    """Smooth one track's field positions under acceleration/speed limits.

    ``samples`` is ``[(frame_index, (x_m, y_m)), ...]`` in any order; the
    result is sorted by frame. Frames separated by more than ``max_gap`` start
    a fresh segment, because nothing sensible can be said about motion across
    a hole -- the first point after a gap is taken as measured.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    ordered = sorted(samples, key=lambda s: s[0])
    if len(ordered) < 2:
        return list(ordered)

    stats = stats if stats is not None else FilterStats()
    out: List[Tuple[int, Point]] = []
    prev_frame: Optional[int] = None
    prev_pos: Optional[Point] = None
    prev_v: Optional[Point] = None

    for frame, measured in ordered:
        stats.points += 1
        if prev_frame is None or frame - prev_frame > max_gap:
            out.append((frame, measured))
            stats.segments += 1
            prev_frame, prev_pos, prev_v = frame, measured, None
            continue

        dt = (frame - prev_frame) / fps
        pos, vel = _limit(prev_pos, prev_v, measured, dt, max_accel, max_speed, stats)
        out.append((frame, pos))
        prev_frame, prev_pos, prev_v = frame, pos, vel

    return out


def filter_tracks(
    tracks: Dict[int, Sequence[Tuple[int, Point]]],
    fps: float = 25.0,
    **kwargs,
) -> Tuple[Dict[int, List[Tuple[int, Point]]], FilterStats]:
    """Apply :func:`filter_track` to every track, sharing one stats record.

    The returned stats are worth logging: a clip where a large share of points
    had to be clamped is telling you the detections or the calibration are
    struggling, not that the players are unusual.
    """
    stats = FilterStats()
    out = {tid: filter_track(s, fps=fps, stats=stats, **kwargs)
           for tid, s in tracks.items()}
    return out, stats
