"""BoT-SORT tracking that consumes our own detections.

Ultralytics normally runs BoT-SORT *inside* ``model.track()``, coupling
detection and tracking. To keep ``YOLODetector`` as the only module that
performs detection, this wrapper drives the standalone
``ultralytics.trackers.BOTSORT`` tracker directly, feeding it our
:class:`Detection` objects through a small ``Boxes``-like adapter.

BoT-SORT is preferred over ByteTrack because its global motion
compensation (GMC) and Kalman model give more stable IDs under camera
motion, occlusion and visually similar players.

Notes on ReID:
    ``reid_model: "auto"`` is the one value that cannot work here -- it means
    "reuse the detector's backbone features", which are only exposed by a
    forward hook on the ``model.track()`` path. Every other value is a real
    encoder: Ultralytics builds it with ``build_encoder`` and it crops from
    the ``frame`` we already hand to ``update()``, so no detector hook is
    involved. Point ``reid_model`` at any YOLO checkpoint (e.g.
    ``weights/yolo11n-cls.pt``) to get appearance ReID;
    with ``with_reid: false`` the tracker falls back to GMC + motion + IoU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from detection.data_models import Detection
from tracking.models import Track
from tracking.tracker import BaseTracker
from utils.config_loader import BotSortParams, TrackingConfig
from utils.logger import get_logger

logger = get_logger("tracking.botsort_tracker")

# Output column layout of BYTETracker/BOTSORT.update():
# [x1, y1, x2, y2, track_id, score, cls, idx]
_COL_TRACK_ID = 4
_COL_SCORE = 5
_COL_CLS = 6

# Minimum IoU for matching an output track back to its source detection.
_SOURCE_IOU_THRESH = 0.5


class _DetectionArray:
    """Adapter exposing a list of detections like an Ultralytics ``Boxes``.

    The tracker only touches ``conf``, ``cls``, ``xywh``, ``xyxy``,
    ``len()`` and boolean-mask indexing, so we implement exactly those.
    Crucially we do NOT define ``xywhr`` — its absence tells the tracker
    these are axis-aligned (not oriented) boxes.
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self._xyxy = xyxy.astype(np.float32).reshape(-1, 4)
        self.conf = conf.astype(np.float32).reshape(-1)
        self.cls = cls.astype(np.float32).reshape(-1)

    @staticmethod
    def from_detections(detections: Sequence[Detection]) -> "_DetectionArray":
        if not detections:
            empty = np.zeros((0, 4), dtype=np.float32)
            return _DetectionArray(empty, np.zeros(0), np.zeros(0))
        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=np.float32)
        return _DetectionArray(xyxy, conf, class_ids)

    @property
    def xyxy(self) -> np.ndarray:
        return self._xyxy

    @property
    def xywh(self) -> np.ndarray:
        ret = self._xyxy.copy()
        ret[:, 2:] -= ret[:, :2]          # width, height
        ret[:, :2] += ret[:, 2:] / 2.0    # center x, y
        return ret

    def __len__(self) -> int:
        return self._xyxy.shape[0]

    def __getitem__(self, index) -> "_DetectionArray":
        return _DetectionArray(self._xyxy[index], self.conf[index], self.cls[index])


def _build_args(
    params: BotSortParams,
    tracker_type: str,
    with_reid: bool,
    reid_model: str,
) -> SimpleNamespace:
    """Assemble the namespace that Ultralytics' BOTSORT expects as ``args``."""
    return SimpleNamespace(
        tracker_type=tracker_type,
        track_high_thresh=params.track_high_thresh,
        track_low_thresh=params.track_low_thresh,
        new_track_thresh=params.new_track_thresh,
        track_buffer=params.track_buffer,
        match_thresh=params.match_thresh,
        fuse_score=params.fuse_score,
        gmc_method=params.gmc_method,
        proximity_thresh=params.proximity_thresh,
        appearance_thresh=params.appearance_thresh,
        with_reid=with_reid,
        model=reid_model,
    )


def _iou(box: Tuple[float, float, float, float], boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against an (N,4) array of xyxy boxes."""
    if boxes.size == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


class BoTSORTTracker(BaseTracker):
    """Tracks players/goalkeepers/referees and (separately) the ball."""

    def __init__(
        self,
        config: TrackingConfig,
        class_names: Dict[int, str],
        frame_rate: int = 30,
        debug: bool = False,
    ) -> None:
        self._config = config
        self._class_names = dict(class_names)
        self._frame_rate = max(int(round(frame_rate)), 1)
        self._ball_class_id = self._resolve_ball_class_id(config.ball_class_name)

        if config.with_reid and config.reid_model == "auto":
            logger.warning(
                "with_reid=true with reid_model='auto' is not supported for a "
                "standalone tracker (native ReID needs model.track()). "
                "Falling back to GMC + motion + IoU. Provide a ReID "
                "checkpoint path to enable appearance ReID."
            )
            effective_reid = False
        else:
            effective_reid = config.with_reid

        self._effective_reid = effective_reid
        self._main_tracker = self._make_tracker(config.main, effective_reid)

        # Ball routing: the custom BallTracker is preferred; the BoT-SORT
        # ball tracker is only a fallback when ball_tracking is disabled.
        self._use_custom_ball = (
            config.ball_tracking.enabled and self._ball_class_id is not None
        )
        self._separate_ball = (
            self._ball_class_id is not None
            and (self._use_custom_ball or config.separate_ball)
        )
        self._ball_tracker = self._build_ball_tracker(debug)
        self._ball_botsort = (
            self._make_tracker(config.ball, effective_reid)
            if self._separate_ball and not self._use_custom_ball
            else None
        )
        logger.info(
            "BoT-SORT ready (tracker=%s, custom_ball=%s, separate_ball=%s, "
            "reid=%s, fps=%d)",
            config.tracker_type,
            self._use_custom_ball,
            self._separate_ball,
            effective_reid,
            self._frame_rate,
        )

    def _build_ball_tracker(self, debug: bool):
        if not self._use_custom_ball:
            return None
        from tracking.ball_tracker import BallTracker

        return BallTracker(
            self._config.ball_tracking,
            self._ball_class_id,
            self._class_names.get(self._ball_class_id, self._config.ball_class_name),
            debug=debug,
        )

    def _make_tracker(self, params: BotSortParams, with_reid: bool):
        from ultralytics.trackers import BOTSORT, BYTETracker  # local heavy import

        args = _build_args(
            params, self._config.tracker_type, with_reid, self._config.reid_model
        )
        tracker_cls = BOTSORT if self._config.tracker_type == "botsort" else BYTETracker
        # Ultralytics has changed the tracker constructor signature across
        # versions: most builds accept ``frame_rate=`` as a kwarg, but some
        # drop it and read it from ``args``. Inspect the signature so we pass
        # frame_rate only when it's actually accepted (and never swallow an
        # unrelated TypeError from inside the constructor).
        import inspect

        params = inspect.signature(tracker_cls.__init__).parameters
        if "frame_rate" in params:
            return tracker_cls(args, frame_rate=self._frame_rate)
        if not hasattr(args, "frame_rate"):
            args.frame_rate = self._frame_rate
        return tracker_cls(args)

    def _resolve_ball_class_id(self, ball_class_name: str) -> Optional[int]:
        for class_id, name in self._class_names.items():
            if name == ball_class_name:
                return class_id
        return None

    # ------------------------------------------------------------------
    # BaseTracker API
    # ------------------------------------------------------------------
    def update(
        self,
        detections: Sequence[Detection],
        frame: np.ndarray,
        frame_index: int,
    ) -> List[Track]:
        # Pair each detection with its index in the frame's detection list,
        # so we can report source_detection_id against the raw detections.
        indexed = list(enumerate(detections))

        if self._separate_ball:
            ball = [(i, d) for i, d in indexed if d.class_id == self._ball_class_id]
            main = [(i, d) for i, d in indexed if d.class_id != self._ball_class_id]
        else:
            ball, main = [], indexed

        tracks = self._run(self._main_tracker, main, frame, frame_index)
        if self._ball_tracker is not None:
            # Custom ball association (handles the zero-IoU small/fast ball).
            tracks.extend(self._ball_tracker.update(ball, frame_index))
        elif self._ball_botsort is not None:
            tracks.extend(self._run(self._ball_botsort, ball, frame, frame_index))
        return tracks

    def reset(self) -> None:
        self._main_tracker.reset()
        if self._ball_tracker is not None:
            self._ball_tracker.reset()
        if self._ball_botsort is not None:
            self._ball_botsort.reset()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run(
        self,
        tracker,
        indexed_dets: List[Tuple[int, Detection]],
        frame: np.ndarray,
        frame_index: int,
    ) -> List[Track]:
        # Always call update (even with no detections) so lost tracks age.
        dets = [d for _, d in indexed_dets]
        adapter = _DetectionArray.from_detections(dets)
        output = tracker.update(adapter, frame)

        if output is None or len(output) == 0:
            return []

        det_boxes = adapter.xyxy
        tracks: List[Track] = []
        for row in output:
            class_id = int(round(row[_COL_CLS]))
            bbox = (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
            )
            source_id = self._match_source(bbox, class_id, indexed_dets, det_boxes)
            tracks.append(
                Track(
                    frame_id=frame_index,
                    track_id=int(round(row[_COL_TRACK_ID])),
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, f"class_{class_id}"),
                    confidence=float(row[_COL_SCORE]),
                    bbox=bbox,
                    source_detection_id=source_id,
                )
            )
        return tracks

    @staticmethod
    def _match_source(
        bbox: Tuple[float, float, float, float],
        class_id: int,
        indexed_dets: List[Tuple[int, Detection]],
        det_boxes: np.ndarray,
    ) -> Optional[int]:
        """Recover the source detection index by best same-class IoU."""
        if not indexed_dets:
            return None
        ious = _iou(bbox, det_boxes)
        best_local = -1
        best_iou = _SOURCE_IOU_THRESH
        for local_idx, (_, det) in enumerate(indexed_dets):
            if det.class_id != class_id:
                continue
            if ious[local_idx] >= best_iou:
                best_iou = ious[local_idx]
                best_local = local_idx
        if best_local < 0:
            return None
        return indexed_dets[best_local][0]
