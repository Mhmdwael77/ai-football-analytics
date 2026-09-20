"""YOLO11-based object detector.

Wraps an Ultralytics YOLO model behind a small, typed interface:
``detect(frame) -> List[Detection]``.  The rest of the pipeline never
touches Ultralytics objects directly, which keeps the detector swappable
and the pipeline testable with a fake model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from detection.data_models import Detection
from utils.config_loader import ModelConfig
from utils.logger import get_logger

logger = get_logger("detection.detector")


class YOLODetector:
    """Detects ball / goalkeeper / player / referee on single frames."""

    def __init__(
        self,
        config: ModelConfig,
        class_names: Dict[int, str],
        model: Optional[object] = None,
    ) -> None:
        """
        Args:
            config: Model section of the application config.
            class_names: Mapping of class index -> class name.
            model: Optional pre-built model (dependency injection for
                tests). When None, the Ultralytics model is loaded from
                ``config.weights_path``.
        """
        self._config = config
        self._class_names = dict(class_names)
        self._device = self._resolve_device(config.device)

        # Inference must run at the LOWEST threshold any class needs;
        # per-class thresholds are applied afterwards in _filter().
        overrides = config.class_confidence_overrides or {}
        self._min_confidence = min(
            [config.confidence_threshold, *overrides.values()]
        )

        if model is not None:
            self._model = model
        else:
            weights = Path(config.weights_path)
            if not weights.is_file():
                raise FileNotFoundError(
                    f"Model weights not found: {weights}. "
                    f"Place your trained 'best.pt' in the weights/ folder "
                    f"or update model.weights_path in the config."
                )
            from ultralytics import YOLO  # heavy import, kept local

            logger.info("Loading YOLO11 weights from %s", weights)
            self._model = YOLO(str(weights))

        self._check_class_names()
        logger.info(
            "Detector ready (device=%s, conf=%.2f, iou=%.2f, imgsz=%d, "
            "overrides=%s)",
            self._device,
            config.confidence_threshold,
            config.iou_threshold,
            config.image_size,
            overrides or "none",
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _check_class_names(self) -> None:
        """Warn if the model's embedded names disagree with the config."""
        model_names = getattr(self._model, "names", None)
        if not isinstance(model_names, dict):
            return
        for class_id, name in self._class_names.items():
            model_name = model_names.get(class_id)
            if model_name is not None and str(model_name) != name:
                logger.warning(
                    "Class %d is '%s' in the config but '%s' in the model "
                    "weights — check your 'classes' mapping.",
                    class_id, name, model_name,
                )

    @property
    def device(self) -> str:
        return self._device

    def warmup(self) -> None:
        """Run one dummy inference so the first real frame isn't slow."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        logger.debug("Warmup inference complete")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame and return typed detections."""
        results = self._model.predict(
            source=frame,
            conf=self._min_confidence,
            iou=self._config.iou_threshold,
            imgsz=self._config.image_size,
            device=self._device,
            verbose=False,
        )
        return self._parse(results[0])

    def detect_with_weak(
        self, frame: np.ndarray
    ) -> Tuple[List[Detection], List[Detection]]:
        """Detect once, return ``(confident, weak)`` from the same inference.

        ``confident`` is exactly what :meth:`detect` returns -- the boxes that
        clear the per-class display thresholds -- and is what gets exported and
        drawn. ``weak`` holds the boxes that survived inference but *failed*
        those thresholds.

        The weak list exists for the tracker. BoT-SORT/ByteTrack associate in
        two passes, and the second pass is over low-score boxes: that is the
        mechanism that carries an id through a partial occlusion, when a
        player's score dips under the display threshold for a few frames.
        Feeding the tracker only the confident boxes leaves that pass with
        nothing to match, so the track dies and the player returns as a NEW id.
        Measured on this project's clips, dropouts like that -- not association
        errors -- cause essentially every id switch.

        Weak boxes never reach the JSON or the video; ``new_track_thresh``
        keeps them from spawning tracks of their own.
        """
        results = self._model.predict(
            source=frame,
            conf=self._min_confidence,
            iou=self._config.iou_threshold,
            imgsz=self._config.image_size,
            device=self._device,
            verbose=False,
        )
        every = self._parse(results[0], apply_thresholds=False)
        confident, weak = [], []
        for det in every:
            if self._passes_threshold(det.class_name, det.confidence):
                confident.append(det)
            else:
                weak.append(det)
        return confident, weak

    def _parse(self, result, apply_thresholds: bool = True) -> List[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        detections: List[Detection] = []
        for box, confidence, class_id in zip(xyxy, confidences, class_ids):
            class_name = self._class_names.get(class_id, f"class_{class_id}")
            if apply_thresholds and not self._passes_threshold(
                class_name, float(confidence)
            ):
                continue
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=tuple(float(v) for v in box),
                )
            )
        return detections

    def _passes_threshold(self, class_name: str, confidence: float) -> bool:
        threshold = self._config.class_confidence_overrides.get(
            class_name, self._config.confidence_threshold
        )
        return confidence >= threshold
