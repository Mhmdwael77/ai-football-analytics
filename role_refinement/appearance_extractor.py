"""Per-track appearance extraction for role refinement.

For every track we sample several frames across its lifetime, crop the
bbox, isolate the upper-body / jersey region, suppress grass/background,
and turn what's left into a normalized HSV histogram. Averaging those
per-frame histograms gives a stable appearance vector per track that the
team clusterer and role refiner reason over.

Design notes:
    * No fixed color rules. Grass suppression masks *green-dominant*
      pixels only as generic background removal — it never maps a color
      to a role or team. Team colors are discovered downstream by
      clustering, not assumed here.
    * Crops are obtained through a :class:`CropProvider`, so production
      (video-backed) and tests (synthetic solid-color crops) share the
      exact same feature code.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from tracking.models import Track
from utils.config_loader import RoleRefinementConfig
from utils.logger import get_logger

logger = get_logger("role_refinement.appearance_extractor")

# Histogram resolution. Hue carries team/kit color; saturation separates
# vivid kits from washed-out / dark (e.g. referee black) ones; value
# separates light from dark. These are descriptive, not prescriptive.
_HUE_BINS = 18      # 10 degrees per bin over OpenCV's 0..180 hue range
_SAT_BINS = 8
_VAL_BINS = 4
FEATURE_DIM = _HUE_BINS + _SAT_BINS + _VAL_BINS

# Each sub-histogram is L1-normalized independently (so a kit isn't judged
# by how many pixels happened to survive masking) and then weighted. Hue
# is the strongest team-color cue, so it gets the most weight; saturation
# and value still matter because they're what makes a dull/dark referee
# kit read as an outlier rather than as a washed-out team color. The hue
# histogram is *saturation-weighted* so near-gray pixels (whose hue is
# meaningless) don't inject phantom color.
_HUE_WEIGHT = 2.0
_SAT_WEIGHT = 1.0
_VAL_WEIGHT = 1.0

# Grass mask (OpenCV HSV ranges: H in 0..180, S/V in 0..255). Anything
# green-ish and reasonably saturated/bright is treated as pitch and
# dropped before histogramming.
# Grass sits around hue 45..75. Starting the band at 35 also deleted the low
# end of a yellow-green referee shirt, so an official was measured mostly from
# whatever showed through and came out a brown that sat close to a kit.
# Measured on Tesr1, as a fraction of the distance between the two kits:
#   hue >= 35  officials 0.24 from the nearer kit, kits 194 apart
#   hue >= 40  officials 0.57 from the nearer kit, kits 188 apart
#   hue >= 45  officials 0.64 from the nearer kit, kits 148 apart
# 40 more than doubles the margin that keeps officials out of a squad and
# costs almost nothing between the kits; 45 buys a little more and starts
# collapsing the two kits into each other, which is the thing that must not
# blur.
_GRASS_HUE_LOW = 40
_GRASS_HUE_HIGH = 85
_GRASS_SAT_MIN = 40
_GRASS_VAL_MIN = 40
# If grass removal leaves too little signal, fall back to all pixels.
_MIN_FOREGROUND_PIXELS = 12


class CropProvider(Protocol):
    """Supplies the BGR bbox crop for a given (track_id, frame_id)."""

    def get(
        self, track_id: int, frame_id: int, bbox: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        ...


class DictCropProvider:
    """In-memory crop provider keyed by ``(track_id, frame_id)``.

    Handy for tests: feed it synthetic solid-color crops and the rest of
    the pipeline behaves exactly as it would on real video.
    """

    def __init__(self, crops: Dict[Tuple[int, int], np.ndarray]) -> None:
        self._crops = crops

    def get(self, track_id, frame_id, bbox):  # noqa: D401 - simple lookup
        return self._crops.get((track_id, frame_id))


class AppearanceExtractor:
    """Builds a stable appearance vector per track."""

    def __init__(self, config: RoleRefinementConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Sampling plan
    # ------------------------------------------------------------------
    def sample_frames(
        self, track_history: Dict[int, List[Track]]
    ) -> Dict[int, List[int]]:
        """Pick up to ``samples_per_track`` frame ids, evenly spaced, per track.

        Returns a mapping ``track_id -> [frame_id, ...]``. The spacing is
        even across the track's lifetime so the appearance vector isn't
        dominated by one moment (e.g. a single occlusion).
        """
        plan: Dict[int, List[int]] = {}
        k = max(self._config.samples_per_track, 1)
        for track_id, observations in track_history.items():
            frame_ids = [obs.frame_id for obs in observations]
            if not frame_ids:
                continue
            if len(frame_ids) <= k:
                plan[track_id] = list(frame_ids)
            else:
                idx = np.linspace(0, len(frame_ids) - 1, k)
                idx = sorted({int(round(i)) for i in idx})
                plan[track_id] = [frame_ids[i] for i in idx]
        return plan

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract(
        self,
        track_history: Dict[int, List[Track]],
        sample_plan: Dict[int, List[int]],
        crop_provider: CropProvider,
        frame_size: Tuple[int, int] = (1, 1),
    ) -> Dict[int, "TrackAppearance"]:
        """Build a :class:`TrackAppearance` for every track.

        Ball tracks get no appearance vector (they are handled by role
        pass-through), but their trajectory summary is still recorded.
        """
        from role_refinement.role_models import TrackAppearance

        results: Dict[int, TrackAppearance] = {}
        ball_name = self._config.ball_class_name

        for track_id, observations in track_history.items():
            detected_class = observations[0].class_name if observations else "unknown"
            centers = [obs.center for obs in observations]
            track_length = len(observations)

            sample_vectors: List[np.ndarray] = []
            if detected_class != ball_name:
                bbox_by_frame = {obs.frame_id: obs.bbox for obs in observations}
                for frame_id in sample_plan.get(track_id, []):
                    bbox = bbox_by_frame.get(frame_id)
                    if bbox is None:
                        continue
                    crop = crop_provider.get(track_id, frame_id, bbox)
                    vec = self.vector_from_crop(crop)
                    if vec is not None:
                        sample_vectors.append(vec)

            if sample_vectors:
                stacked = np.stack(sample_vectors, axis=0)
                mean_vec = stacked.mean(axis=0)
                mean_vec = _l1_normalize(mean_vec)
            else:
                stacked = np.empty((0, FEATURE_DIM), dtype=np.float32)
                mean_vec = np.empty((0,), dtype=np.float32)

            results[track_id] = TrackAppearance(
                track_id=track_id,
                detected_class=detected_class,
                vector=mean_vec,
                sample_vectors=stacked,
                n_samples=len(sample_vectors),
                track_length=track_length,
                centers=centers,
                frame_size=frame_size,
            )
        return results

    # ------------------------------------------------------------------
    # Single-crop feature
    # ------------------------------------------------------------------
    def vector_from_crop(self, crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """HSV-histogram feature for one BGR crop, or ``None`` if unusable."""
        if crop is None or crop.size == 0:
            return None
        if crop.ndim != 3 or crop.shape[2] != 3:
            return None

        jersey = self._jersey_region(crop)
        if jersey.size == 0:
            return None

        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0].reshape(-1)
        s = hsv[..., 1].reshape(-1)
        v = hsv[..., 2].reshape(-1)

        mask = self._foreground_mask(h, s, v)
        if mask.sum() < _MIN_FOREGROUND_PIXELS:
            mask = np.ones_like(h, dtype=bool)  # grass removal too aggressive

        hist = self._hsv_histogram(h[mask], s[mask], v[mask])
        if hist is None:
            return None
        return hist

    def median_bgr(self, crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Median jersey colour in raw BGR, grass removed, or ``None``.

        The histogram feature above is built for asking whether two shirts are
        the same, and it is normalised and hue-weighted to make that robust.
        That normalisation is exactly wrong for the other question -- how many
        *kinds* of shirt are on this pitch -- because it flattens the very
        difference that separates them. Measured on Tesr1, clustering the
        histograms put the referee in a team and gave him a squad number;
        clustering these medians left him out, because his yellow sits 160
        apart from either kit in blue-versus-green while the two kits sit 40
        apart from each other in the same terms.

        So: use the histogram to compare shirts, and this to count them.
        """
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return None
        jersey = self._jersey_region(crop)
        if jersey.size == 0:
            return None
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        flat = jersey.reshape(-1, 3)
        mask = self._foreground_mask(hsv[:, 0], hsv[:, 1], hsv[:, 2])
        if mask.sum() < _MIN_FOREGROUND_PIXELS:
            return None            # nothing but grass: no opinion, not a guess
        return np.median(flat[mask].astype(np.float32), axis=0)

    def dominant_hsv(self, crop: Optional[np.ndarray]) -> Optional[Tuple[int, int, int]]:
        """Approximate dominant jersey HSV color (for reasons/debugging)."""
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return None
        jersey = self._jersey_region(crop)
        if jersey.size == 0:
            return None
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        mask = self._foreground_mask(h, s, v)
        if mask.sum() < _MIN_FOREGROUND_PIXELS:
            mask = np.ones_like(h, dtype=bool)
        sel = hsv[mask]
        return (
            int(np.median(sel[:, 0])),
            int(np.median(sel[:, 1])),
            int(np.median(sel[:, 2])),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _jersey_region(self, crop: np.ndarray) -> np.ndarray:
        """Crop to the upper-body band and trim the sides (arms/background)."""
        h, w = crop.shape[:2]
        top = int(round(self._config.jersey_crop_top * h))
        bottom = int(round(self._config.jersey_crop_bottom * h))
        side = int(round(self._config.jersey_crop_side * w))
        top = min(max(top, 0), h)
        bottom = min(max(bottom, top + 1), h)
        left = min(max(side, 0), w)
        right = max(min(w - side, w), left + 1)
        return crop[top:bottom, left:right]

    @staticmethod
    def _foreground_mask(
        h: np.ndarray, s: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """True where a pixel is NOT grass (kept for histogramming)."""
        grass = (
            (h >= _GRASS_HUE_LOW)
            & (h <= _GRASS_HUE_HIGH)
            & (s >= _GRASS_SAT_MIN)
            & (v >= _GRASS_VAL_MIN)
        )
        return ~grass

    @staticmethod
    def _hsv_histogram(
        h: np.ndarray, s: np.ndarray, v: np.ndarray
    ) -> Optional[np.ndarray]:
        if h.size == 0:
            return None
        # Saturation-weighted hue: gray/dark pixels contribute ~no hue mass,
        # so a dark referee kit shows up as low-saturation rather than as
        # some arbitrary hue. The weight is the pixel's saturation in [0, 1].
        sat_weight = s.astype(np.float32) / 255.0
        h_hist, _ = np.histogram(h, bins=_HUE_BINS, range=(0, 180), weights=sat_weight)
        s_hist, _ = np.histogram(s, bins=_SAT_BINS, range=(0, 256))
        v_hist, _ = np.histogram(v, bins=_VAL_BINS, range=(0, 256))
        feature = np.concatenate(
            [
                _HUE_WEIGHT * _l1_normalize(h_hist.astype(np.float32)),
                _SAT_WEIGHT * _l1_normalize(s_hist.astype(np.float32)),
                _VAL_WEIGHT * _l1_normalize(v_hist.astype(np.float32)),
            ]
        ).astype(np.float32)
        # Tidy global L1 (cosine-invariant; keeps the vector summing to ~1).
        return _l1_normalize(feature)


def _l1_normalize(vec: np.ndarray) -> np.ndarray:
    total = float(vec.sum())
    if total <= 0:
        return vec.astype(np.float32)
    return (vec / total).astype(np.float32)


def build_track_history(frames: Sequence) -> Dict[int, List[Track]]:
    """Group a flat sequence of per-frame :class:`FrameTracks` by track_id.

    ``frames`` is any iterable of objects exposing ``.tracks`` (e.g.
    :class:`tracking.models.FrameTracks`). Returns an insertion-ordered
    mapping ``track_id -> [Track, ...]`` sorted by frame.
    """
    history: Dict[int, List[Track]] = {}
    for frame in frames:
        for track in frame.tracks:
            history.setdefault(track.track_id, []).append(track)
    for observations in history.values():
        observations.sort(key=lambda t: t.frame_id)
    return history
