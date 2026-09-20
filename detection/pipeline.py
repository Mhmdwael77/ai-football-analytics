"""Pipeline: video in -> detections (-> tracks) -> annotated video + JSON.

The pipeline only orchestrates; all real work lives in the injected
components (detector, tracker, annotators, exporters, video I/O), so each
piece can be tested in isolation and both the detector and tracker can be
replaced with fakes.

Two modes:
    * Detection only  — Phase 1 behaviour, unchanged.
    * Detection + Tracking — when ``config.tracking.enabled`` (or a tracker
      is injected). Detection still runs and its JSON is still exported;
      tracking is layered on top and writes a separate tracks JSON.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2

from detection.detector import YOLODetector
from tracking.tracker import BaseTracker
from utils.config_loader import AppConfig
from utils.fps_meter import FPSMeter
from utils.json_exporter import DetectionJSONExporter
from utils.logger import get_logger
from utils.video_io import VideoReader, VideoWriter
from visualization.annotator import DetectionAnnotator

logger = get_logger("detection.pipeline")

_PROGRESS_EVERY = 100  # log progress every N frames
_WINDOW_NAME = "football_ai"


class DetectionPipeline:
    """Runs detection (and optionally tracking) over a full video."""

    def __init__(
        self,
        config: AppConfig,
        detector: Optional[YOLODetector] = None,
        tracker: Optional[BaseTracker] = None,
    ) -> None:
        """
        Args:
            config: Full application configuration.
            detector: Optional pre-built detector (DI for tests). When
                None, a YOLODetector is created from the config.
            tracker: Optional pre-built tracker (DI for tests). When None
                and tracking is enabled, a BoTSORTTracker is built lazily
                from the config (using the video's frame rate).
        """
        self._config = config
        self._detector = detector or YOLODetector(config.model, config.classes)
        self._annotator = DetectionAnnotator(
            config.visualization, debug=config.debug.enabled
        )
        self._injected_tracker = tracker
        self._tracking_active = tracker is not None or config.tracking.enabled

    def run(self, video_path: str | Path, max_frames: Optional[int] = None) -> Dict:
        """Process one video end to end and return a run summary."""
        video_path = Path(video_path)
        output_dir = Path(self._config.video.output_dir)
        stem = video_path.stem

        annotated_path = output_dir / f"{stem}_annotated.mp4"
        detections_json = output_dir / f"{stem}_detections.json"
        tracks_json = self._tracks_json_path(output_dir, stem)
        debug_dir = output_dir / "debug_frames" / stem

        phase = "1+2 detection+tracking" if self._tracking_active else "1 detection"
        logger.info("=== Phase %s run: %s ===", phase, video_path.name)
        self._detector.warmup()

        fps_meter = FPSMeter()
        det_class_totals: Dict[str, int] = {}
        trk_class_totals: Dict[str, int] = {}
        stopped_early = False

        with VideoReader(video_path) as reader:
            tracker = self._resolve_tracker(reader.fps)
            track_visualizer = self._build_track_visualizer() if tracker else None

            det_exporter = DetectionJSONExporter(
                detections_json,
                metadata={
                    "video": str(video_path),
                    "weights": self._config.model.weights_path,
                    "confidence_threshold": self._config.model.confidence_threshold,
                    "iou_threshold": self._config.model.iou_threshold,
                    "image_size": self._config.model.image_size,
                    "video_fps": round(reader.fps, 3),
                    "resolution": [reader.width, reader.height],
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            trk_exporter = self._build_track_exporter(
                tracks_json, video_path, reader
            ) if tracker else None

            writer: Optional[VideoWriter] = None
            if self._config.video.save_video:
                writer = VideoWriter(
                    annotated_path,
                    fps=reader.fps,
                    frame_size=(reader.width, reader.height),
                    codec=self._config.video.output_codec,
                )

            try:
                for frame_index, frame in reader.frames():
                    fps_meter.start()
                    if tracker and self._config.tracking.feed_weak_detections:
                        # One inference, two lists: the confident boxes are what
                        # we export and draw; the weak ones exist only so the
                        # tracker's low-score association pass can hold an id
                        # through a dropout instead of minting a new one.
                        detections, weak = self._detector.detect_with_weak(frame)
                        tracks = tracker.update(
                            detections + weak, frame, frame_index
                        )
                    else:
                        detections = self._detector.detect(frame)
                        tracks = (
                            tracker.update(detections, frame, frame_index)
                            if tracker
                            else None
                        )
                    fps_meter.stop()

                    # Raw detections are always exported (preserved for debug).
                    for det in detections:
                        det_class_totals[det.class_name] = (
                            det_class_totals.get(det.class_name, 0) + 1
                        )
                    det_exporter.add_frame(frame_index, detections)

                    if tracker:
                        for trk in tracks:
                            trk_class_totals[trk.class_name] = (
                                trk_class_totals.get(trk.class_name, 0) + 1
                            )
                        if trk_exporter:
                            trk_exporter.add_frame(frame_index, tracks)
                        annotated = track_visualizer.annotate(
                            frame, tracks,
                            frame_index=frame_index, fps=fps_meter.current_fps,
                        )
                    else:
                        annotated = self._annotator.annotate(
                            frame, detections,
                            frame_index=frame_index, fps=fps_meter.current_fps,
                        )

                    if writer is not None:
                        writer.write(annotated)

                    if self._should_save_debug_frame(frame_index):
                        self._save_debug_frame(debug_dir, frame_index, annotated)

                    if self._config.debug.show_window:
                        cv2.imshow(_WINDOW_NAME, annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logger.warning("Stopped by user at frame %d", frame_index)
                            stopped_early = True
                            break

                    if frame_index % _PROGRESS_EVERY == 0:
                        self._log_progress(frame_index, reader, fps_meter)

                    if max_frames is not None and frame_index >= max_frames:
                        logger.info("Reached max_frames=%d, stopping", max_frames)
                        break
            finally:
                if writer is not None:
                    writer.release()
                if self._config.debug.show_window:
                    cv2.destroyAllWindows()

        det_exporter.save()
        if trk_exporter:
            trk_exporter.save()

        summary = {
            "video": str(video_path),
            "tracking": bool(tracker),
            "frames_processed": fps_meter.frame_count,
            "stopped_early": stopped_early,
            "total_detections": det_exporter.total_detections,
            "detections_by_class": det_class_totals,
            "average_fps": round(fps_meter.average_fps, 2),
            "inference_seconds": round(fps_meter.total_time, 2),
            "annotated_video": str(annotated_path)
            if self._config.video.save_video
            else None,
            "detections_json": str(detections_json),
        }
        if tracker:
            summary["tracks_by_class"] = trk_class_totals
            summary["total_tracks"] = trk_exporter.total_tracks if trk_exporter else 0
            summary["unique_track_ids"] = (
                trk_exporter.unique_track_count if trk_exporter else 0
            )
            summary["tracks_json"] = str(tracks_json) if trk_exporter else None
        logger.info("Run complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Tracking helpers
    # ------------------------------------------------------------------
    def _resolve_tracker(self, frame_rate: float) -> Optional[BaseTracker]:
        if self._injected_tracker is not None:
            return self._injected_tracker
        if not self._config.tracking.enabled:
            return None
        # Heavy import kept local so detection-only runs never need it.
        from tracking.botsort_tracker import BoTSORTTracker

        return BoTSORTTracker(
            self._config.tracking,
            self._config.classes,
            frame_rate=frame_rate,
            debug=self._config.debug.enabled,
        )

    def _build_track_visualizer(self):
        from tracking.track_visualizer import TrackVisualizer

        return TrackVisualizer(
            self._config.visualization, debug=self._config.debug.enabled
        )

    def _build_track_exporter(self, tracks_json, video_path, reader):
        if not self._config.tracking.save_tracks:
            return None
        from tracking.track_exporter import TrackJSONExporter

        return TrackJSONExporter(
            tracks_json,
            metadata={
                "tracker": self._config.tracking.tracker_type,
                "weights": self._config.model.weights_path,
                "separate_ball": self._config.tracking.separate_ball,
                "with_reid": self._config.tracking.with_reid,
                "video_fps": round(reader.fps, 3),
                "resolution": [reader.width, reader.height],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

    def _tracks_json_path(self, output_dir: Path, stem: str) -> Path:
        override = self._config.tracking.output_path
        if override:
            return Path(override)
        return output_dir / f"{stem}_tracks.json"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _should_save_debug_frame(self, frame_index: int) -> bool:
        debug = self._config.debug
        return (
            debug.enabled
            and debug.save_debug_frames
            and frame_index % max(debug.debug_frame_interval, 1) == 0
        )

    @staticmethod
    def _save_debug_frame(debug_dir: Path, frame_index: int, frame) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(path), frame)
        logger.debug("Saved debug frame %s", path)

    @staticmethod
    def _log_progress(
        frame_index: int, reader: VideoReader, fps_meter: FPSMeter
    ) -> None:
        if reader.frame_count:
            percent = 100.0 * frame_index / reader.frame_count
            logger.info(
                "Frame %d/%d (%.1f%%) — %.1f fps",
                frame_index, reader.frame_count, percent, fps_meter.average_fps,
            )
        else:
            logger.info("Frame %d — %.1f fps", frame_index, fps_meter.average_fps)
