"""football_ai — Phase 1 (Detection) entry point.

Examples:
    # Basic run with the default config
    python main.py --video data/match.mp4

    # Override confidence threshold and force GPU
    python main.py --video data/match.mp4 --conf 0.4 --device cuda:0

    # Visual debugging: HUD overlay + live window + saved debug frames
    python main.py --video data/match.mp4 --debug --display

    # JSON only, skip writing the annotated video
    python main.py --video data/match.mp4 --no-save-video
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from detection.pipeline import DetectionPipeline
from utils.config_loader import AppConfig, load_config
from utils.logger import get_logger, setup_logging



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="football_ai",
        description="Phase 1: detect ball, goalkeeper, player and referee "
        "on football video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the WHOLE project in one go (tracking + stitching + roles + "
        "corrections + calibration + minimap overlay). Missing human-input "
        "steps (pitch calibration, role review) are offered interactively.",
    )
    parser.add_argument("--weights", help="Override model.weights_path")
    parser.add_argument(
        "--conf", type=float, help="Override model.confidence_threshold"
    )
    parser.add_argument("--device", help="Override model.device (cpu / cuda:0)")
    parser.add_argument("--output-dir", help="Override video.output_dir")
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Skip writing the annotated output video",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Write the annotated output video (on by default; explicit opt-in)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable visual debugging (HUD overlay + saved debug frames)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live preview window (press 'q' to stop)",
    )
    parser.add_argument(
        "--log-level", help="Override logging.level (DEBUG / INFO / ...)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process at most N frames (quick validation on long videos)",
    )

    tracking = parser.add_argument_group("tracking (Phase 2)")
    tracking.add_argument(
        "--tracking",
        action="store_true",
        help="Enable BoT-SORT tracking on top of detection",
    )
    tracking.add_argument(
        "--tracker",
        choices=["botsort", "bytetrack"],
        help="Tracker backend (default: botsort)",
    )
    tracking.add_argument(
        "--save-tracks",
        action="store_true",
        help="Write the per-frame tracking JSON (on by default when tracking)",
    )
    tracking.add_argument(
        "--tracking-output",
        help="Override the tracking JSON output path",
    )
    tracking.add_argument(
        "--stitch-tracks",
        action="store_true",
        help="Re-link broken track IDs after tracking (reduces ID switches)",
    )
    tracking.add_argument(
        "--track-buffer",
        type=int,
        help="Frames a lost track is kept alive for re-association inside "
        "BoT-SORT (default 30; try 60 to cut ID switches at the source)",
    )
    tracking.add_argument(
        "--fix-swaps",
        action="store_true",
        help="Fix ID swaps from crossing players using jersey appearance",
    )
    tracking.add_argument(
        "--reuse-tracks",
        action="store_true",
        help="Reuse the tracks from a previous run (skip detection/tracking) so "
        "track IDs stay identical -- run this after labelling teams so your "
        "manual labels still apply",
    )

    roles = parser.add_argument_group("role refinement (Phase 3)")
    roles.add_argument(
        "--role-refinement",
        action="store_true",
        help="Refine each track's role after tracking (implies --tracking)",
    )
    roles.add_argument(
        "--roles-output",
        help="Override the roles JSON output path",
    )
    roles.add_argument(
        "--save-role-video",
        action="store_true",
        help="Render an annotated role video (outputs/<video>_roles.mp4)",
    )

    manual = parser.add_argument_group("manual role correction (Phase 3.5)")
    manual.add_argument(
        "--manual-role-review",
        action="store_true",
        help="Run the mandatory review/name-entry UI "
        "(implies --role-refinement)",
    )
    manual.add_argument(
        "--apply-role-corrections",
        nargs="?",
        const="__default__",
        default=None,
        metavar="CORRECTIONS_JSON",
        help="Apply manual corrections and write <video>_roles_final.json "
        "(optionally pass the corrections file path)",
    )
    manual.add_argument(
        "--corrections-file",
        help="Path to the manual corrections JSON (read by review + apply)",
    )
    manual.add_argument(
        "--save-final-video",
        action="store_true",
        help="Render the final annotated role video (with USER tags)",
    )

    calib = parser.add_argument_group("pitch calibration / minimap (Phase 4)")
    calib.add_argument(
        "--calibration",
        help="Path to a pitch_calibration.json (enables field projection)",
    )
    calib.add_argument(
        "--auto-calibration",
        action="store_true",
        help="Auto-calibrate per frame from the pitch-keypoint model "
        "(no manual clicking; follows the camera)",
    )
    calib.add_argument(
        "--keypoint-model",
        help="Path to the pitch-keypoint YOLO-pose model (for --auto-calibration)",
    )
    calib.add_argument(
        "--keypoint-backend",
        choices=["yolo_pose", "nbjw"],
        help="Auto-calibration backend: yolo_pose (28-kp, default) or nbjw "
        "(57-kp HRNet, more accurate on broadcast)",
    )
    calib.add_argument(
        "--save-minimap",
        action="store_true",
        help="Render the top-down minimap video (needs --calibration)",
    )
    calib.add_argument(
        "--minimap-overlay",
        action="store_true",
        help="Composite the minimap into the main video "
        "(outputs/<video>_final_with_minimap.mp4); implies --save-minimap",
    )

    analytics = parser.add_argument_group("analytics (Phase 6)")
    analytics.add_argument(
        "--possession",
        action="store_true",
        help="Compute per-team ball possession (image-space, no calibration)",
    )
    analytics.add_argument(
        "--save-analytics",
        action="store_true",
        help="Compute speed & distance from field metres (Phase 5; needs the "
        "minimap field positions)",
    )
    analytics.add_argument(
        "--show-speed-overlay",
        action="store_true",
        help="Draw each player's speed (#id | km/h) on the final video",
    )
    analytics.add_argument(
        "--pose-skeleton",
        action="store_true",
        help="Draw team-colored player pose skeletons on the final video",
    )
    analytics.add_argument(
        "--pose-model",
        help="Path to a YOLO-pose model (default: yolo11m-pose.pt, auto-downloads)",
    )
    calib.add_argument(
        "--minimap-color-mode",
        choices=["team_role", "track_id"],
        help="Minimap dot coloring: team_role (default) or track_id (debug)",
    )
    calib.add_argument(
        "--minimap-show-track-ids",
        action="store_true",
        help="Draw track ids next to minimap dots",
    )
    return parser.parse_args(argv)


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Command-line flags take precedence over the YAML file."""
    # --all turns on the whole project. It's expressed by flipping the
    # individual flags so the rest of this function wires everything up
    # exactly as if the user had passed them all.
    if args.all:
        args.tracking = True
        args.stitch_tracks = True
        args.fix_swaps = True
        args.role_refinement = True
        args.save_video = True
        args.minimap_overlay = True
        args.possession = True
        args.save_analytics = True        # Phase 5 speed & distance
        args.pose_skeleton = True        # draw skeletons instead of plain boxes
        # Prefer automatic calibration when the keypoint model is present
        # (zero manual clicking); otherwise fall back to the manual file.
        if not args.auto_calibration and not args.calibration:
            if Path(config.calibration.keypoint_model).is_file():
                args.auto_calibration = True
            elif config.calibration.calibration_path:
                args.calibration = config.calibration.calibration_path
            else:
                stem = Path(args.video).stem
                out = config.video.output_dir.rstrip("/\\")
                args.calibration = f"{out}/{stem}_pitch_calibration.json"
        # Apply-mode for corrections (uses the saved file if present).
        if args.apply_role_corrections is None:
            args.apply_role_corrections = "__default__"

    if args.weights:
        config.model.weights_path = args.weights
    if args.conf is not None:
        if not 0.0 <= args.conf <= 1.0:
            raise ValueError(f"--conf must be in [0, 1], got {args.conf}")
        config.model.confidence_threshold = args.conf
    if args.device:
        config.model.device = args.device
    if args.output_dir:
        config.video.output_dir = args.output_dir
    if args.save_video:
        config.video.save_video = True
    if args.no_save_video:
        config.video.save_video = False
    if args.debug:
        config.debug.enabled = True
        config.debug.save_debug_frames = True
    if args.display:
        config.debug.enabled = True
        config.debug.show_window = True
    if args.log_level:
        config.logging.level = args.log_level

    # Tracking flags
    if args.tracking:
        config.tracking.enabled = True
    if args.tracker:
        config.tracking.tracker_type = args.tracker
        config.tracking.enabled = True
    if args.save_tracks:
        config.tracking.save_tracks = True
        config.tracking.enabled = True
    if args.tracking_output:
        config.tracking.output_path = args.tracking_output
        config.tracking.enabled = True
    if args.stitch_tracks:
        config.tracking.stitching.enabled = True
        config.tracking.enabled = True
        config.tracking.save_tracks = True
    if args.fix_swaps:
        config.tracking.swap_correction.enabled = True
        config.tracking.enabled = True
        config.tracking.save_tracks = True
    if args.track_buffer is not None:
        if args.track_buffer <= 0:
            raise ValueError(f"--track-buffer must be positive, got {args.track_buffer}")
        config.tracking.main.track_buffer = args.track_buffer
        config.tracking.enabled = True

    # Role refinement (Phase 3) runs on top of tracking, so it implies
    # tracking + a tracks JSON to consume.
    if args.role_refinement:
        config.role_refinement.enabled = True
    if args.roles_output:
        config.role_refinement.output_path = args.roles_output
        config.role_refinement.enabled = True
    if args.save_role_video:
        config.role_refinement.save_role_video = True
        config.role_refinement.enabled = True

    # Manual role correction (Phase 3.5) sits on top of role refinement.
    mc = config.manual_correction
    if args.corrections_file:
        mc.corrections_file = args.corrections_file
    if args.apply_role_corrections is not None:
        if args.apply_role_corrections != "__default__":
            mc.corrections_file = args.apply_role_corrections
        mc.enabled = True
        config.role_refinement.enabled = True
    if args.manual_role_review:
        mc.enabled = True
        config.role_refinement.enabled = True
    if args.save_final_video:
        mc.save_final_video = True

    # Pitch calibration / minimap (Phase 4): needs tracking output to project.
    if args.calibration:
        config.calibration.calibration_path = args.calibration
    if args.save_minimap:
        config.calibration.save_minimap = True
    if args.auto_calibration:
        config.calibration.auto = True
    if args.keypoint_model:
        config.calibration.keypoint_model = args.keypoint_model
    if args.keypoint_backend:
        config.calibration.keypoint_backend = args.keypoint_backend
    if args.minimap_overlay:
        # Overlay implies producing the minimap in the first place.
        config.minimap_overlay.enabled = True
        config.calibration.save_minimap = True
    if args.possession:
        config.possession.enabled = True
        config.tracking.enabled = True
        config.tracking.save_tracks = True
        config.role_refinement.enabled = True
    if args.save_analytics or args.show_speed_overlay:
        # Speed/distance need the minimap field positions (metres).
        config.analytics.enabled = True
        config.calibration.save_minimap = True
        config.tracking.enabled = True
        config.tracking.save_tracks = True
    if args.show_speed_overlay:
        config.analytics.show_speed_overlay = True
        config.minimap_overlay.enabled = True   # the speed labels ride the final video
    if args.pose_skeleton:
        config.pose.enabled = True
        config.minimap_overlay.enabled = True   # skeletons render on the final video
        config.calibration.save_minimap = True
    if args.pose_model:
        config.pose.model = args.pose_model
        config.pose.enabled = True
    if args.minimap_color_mode:
        config.minimap.color_mode = args.minimap_color_mode
    if args.minimap_show_track_ids:
        config.minimap.show_track_ids = True
    if (config.calibration.calibration_path or config.calibration.save_minimap
            or config.calibration.auto):
        config.tracking.enabled = True
        config.tracking.save_tracks = True

    role_outputs_requested = (
        config.role_refinement.enabled
        or config.possession.enabled
        or config.minimap_overlay.enabled
        or config.calibration.save_minimap
        or config.calibration.calibration_path
        or config.calibration.auto
    )
    if role_outputs_requested:
        config.role_refinement.enabled = True
        mc.enabled = True

    if config.role_refinement.enabled:
        config.tracking.enabled = True
        config.tracking.save_tracks = True

    _resolve_per_video_paths(config, args)
    return config


def _resolve_per_video_paths(config, args) -> None:
    """Make human-input file defaults per-video so testing another video
    doesn't reuse the wrong corrections/labels. Only fixed *default* names
    are made stem-based; anything the user set explicitly is respected."""
    from utils.config_loader import ManualCorrectionConfig

    stem = Path(args.video).stem
    out = config.video.output_dir.rstrip("/\\")
    defaults = ManualCorrectionConfig()
    mc = config.manual_correction

    if args.corrections_file is None and \
            mc.corrections_file == defaults.corrections_file:
        mc.corrections_file = f"{out}/{stem}_role_corrections.json"
    if mc.initialization_labels_file == defaults.initialization_labels_file:
        mc.initialization_labels_file = f"{out}/{stem}_init_labels.json"

    cal = config.calibration
    if (not args.calibration and not cal.auto
            and cal.calibration_path == "outputs/pitch_calibration.json"):
        cal.calibration_path = f"{out}/{stem}_pitch_calibration.json"


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
        config = apply_overrides(config, args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    logger = setup_logging(
        level=config.logging.level,
        log_dir=config.logging.log_dir,
        log_to_file=config.logging.log_to_file,
    )

    video_path = Path(args.video)
    if not video_path.is_file():
        logger.error("Input video not found: %s", video_path)
        return 2

    # Human-in-the-loop step #1 (pitch calibration). Offered up front so the
    # rest of the run is unattended. Skipped when auto-calibration is on
    # (the keypoint model needs no clicking) or when non-interactive.
    if config.calibration.calibration_path and not config.calibration.auto:
        ensure_calibration(config, video_path, logger)

    reused_tracks = _find_reusable_tracks(config, video_path) if args.reuse_tracks else None
    if args.reuse_tracks and reused_tracks is None:
        logger.warning("--reuse-tracks set but no existing tracks JSON found for "
                       "%s; running full detection instead.", Path(video_path).stem)
    if reused_tracks is not None:
        # Reuse the tracks from a previous run so track ids stay identical:
        # this is what lets manual team/referee labels (keyed to ids) still
        # apply after a re-run. Detection, tracking, stitching and swap
        # correction are all skipped -- the reused file is already their output.
        logger.info("Reusing existing tracks %s (skipping detection/tracking)",
                    reused_tracks)
        print("\n=== Reusing tracks (no re-detection) ===")
        print(f"  tracks JSON      : {reused_tracks}")
        summary = {"tracking": True, "tracks_json": reused_tracks, "reused": True}
    else:
        try:
            pipeline = DetectionPipeline(config)
            summary = pipeline.run(video_path, max_frames=args.max_frames)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 2
        except Exception:
            logger.exception("Pipeline failed with an unexpected error")
            return 1

        print("\n=== Detection summary ===")
        print(f"  frames processed : {summary['frames_processed']}")
        print(f"  total detections : {summary['total_detections']}")
        for name, count in sorted(summary["detections_by_class"].items()):
            print(f"    {name:<12}: {count}")
        print(f"  average FPS      : {summary['average_fps']}")
        if summary["annotated_video"]:
            print(f"  annotated video  : {summary['annotated_video']}")
        print(f"  detections JSON  : {summary['detections_json']}")

    if not summary.get("reused") and summary.get("tracking"):
        print("\n=== Tracking summary ===")
        print(f"  unique track IDs : {summary['unique_track_ids']}")
        print(f"  total track rows : {summary['total_tracks']}")
        for name, count in sorted(summary["tracks_by_class"].items()):
            print(f"    {name:<12}: {count}")
        if summary.get("tracks_json"):
            print(f"  tracks JSON      : {summary['tracks_json']}")

    # Post-tracking ID stitching (before role refinement so downstream uses
    # the re-linked, more stable identities). Skipped when reusing tracks --
    # the reused file already went through stitching and swap correction, and
    # re-running them would change ids and break the manual labels.
    if (not summary.get("reused")
            and config.tracking.stitching.enabled and summary.get("tracks_json")):
        try:
            stitch_stats = run_stitching(config, video_path, summary)
        except Exception:
            logger.exception("Track stitching failed (continuing with raw tracks)")
            stitch_stats = None
        if stitch_stats is not None:
            print("\n=== Track stitching ===")
            print(f"  ids before       : {stitch_stats['ids_before']}")
            print(f"  ids after        : {stitch_stats['ids_after']}")
            print(f"  re-linked        : {stitch_stats['merges']}")
            print(f"  stitched JSON    : {summary['tracks_json']}")

    # Appearance ID-swap correction (after stitching, before role refinement).
    if (not summary.get("reused")
            and config.tracking.swap_correction.enabled
            and summary.get("tracks_json")):
        try:
            swap_stats = run_swap_correction(config, video_path, summary)
        except Exception:
            logger.exception("Swap correction failed (continuing)")
            swap_stats = None
        if swap_stats is not None:
            print("\n=== ID-swap correction ===")
            print(f"  swaps fixed      : {swap_stats['swaps']}")
            print(f"  drift splits     : {swap_stats.get('splits', 0)}")
            if swap_stats["pairs"]:
                print(f"  pairs            : {swap_stats['pairs']}")

    # Fixed squad numbers over the corrected tracks, before role refinement so
    # that everything downstream is keyed by a squad number rather than by
    # whatever id the tracker happened to mint.
    if config.tracking.roster.enabled and summary.get("tracks_json"):
        try:
            roster_stats = run_roster(config, video_path, summary)
        except Exception:
            logger.exception("Roster numbering failed (continuing with raw ids)")
            roster_stats = None
        if roster_stats is not None:
            used = roster_stats["slots_used"]
            print("\n=== Squad numbers ===")
            print(f"  numbered         : {roster_stats['numbered']}")
            print(f"  per team         : "
                  f"{used.get(0, 0)}/{config.tracking.roster.slots_per_team}, "
                  f"{used.get(1, 0)}/{config.tracking.roster.slots_per_team}")
            print(f"  left unnumbered  : {roster_stats['overflow']}")
            print(f"  roles from       : "
                  f"{'pitch position' if roster_stats['by_position'] else 'colour only'}")
            print(f"  seen             : {roster_stats['roles']}")
            print(f"  roster JSON      : {summary['tracks_json']}")

    # Manual ID-switch swaps the reviewer marked (track surgery before roles).
    # Runs in reuse mode too -- it re-derives from the stable auto-corrected
    # tracks, so it stays idempotent regardless of the reused file.
    if config.manual_correction.enabled and summary.get("tracks_json"):
        try:
            manual_swap_stats = run_manual_swap(config, video_path, summary)
        except Exception:
            logger.exception("Manual swap correction failed (continuing)")
            manual_swap_stats = None
        if manual_swap_stats is not None:
            print("\n=== Manual ID-switch swaps ===")
            print(f"  swaps applied    : {manual_swap_stats['swaps']}")
            print(f"  pairs            : {manual_swap_stats['pairs']}")
            print(f"  corrected tracks : {summary['tracks_json']}")

    role_summary = None
    if config.role_refinement.enabled:
        try:
            role_summary = run_role_refinement(config, video_path, summary)
        except Exception:
            logger.exception("Role refinement failed")
            return 1
        if role_summary is not None:
            print("\n=== Role refinement summary (Phase 3) ===")
            for name, count in sorted(role_summary["roles_by_type"].items()):
                print(f"    {name:<12}: {count}")
            print(f"  refined tracks   : {role_summary['n_tracks']}")
            print(f"  roles JSON       : {role_summary['roles_json']}")
            if role_summary.get("role_video"):
                print(f"  role video       : {role_summary['role_video']}")

        # Mandatory review/name-entry step before any role-dependent outputs.
        if config.manual_correction.enabled and role_summary is not None:
            try:
                correction_summary = run_manual_correction(
                    config, args, video_path, summary, role_summary
                )
                if correction_summary is not None:
                    role_summary["roles_json"] = correction_summary[
                        "final_roles_json"]
                    role_summary["final_roles_json"] = correction_summary[
                        "final_roles_json"]
            except Exception:
                logger.exception("Manual role correction failed")
                return 1

    # Phase 6: ball possession (image-space; no calibration needed).
    if config.possession.enabled and summary.get("tracks_json"):
        try:
            poss = run_possession(config, video_path, summary, role_summary)
        except Exception:
            logger.exception("Possession computation failed (continuing)")
            poss = None
        if poss is not None:
            print("\n=== Ball possession (Phase 6) ===")
            print(f"    Team 0       : {poss['pct'][0]}%")
            print(f"    Team 1       : {poss['pct'][1]}%")
            print(f"  possession JSON  : {poss['json']}")

    # Phase 4: pitch calibration / minimap. Never crash the pipeline.
    if (config.calibration.calibration_path
            or config.calibration.save_minimap
            or config.calibration.auto
            or config.minimap_overlay.enabled):
        try:
            minimap_summary = run_minimap(config, video_path, summary, role_summary)
        except Exception:
            logger.exception("Minimap generation failed (continuing)")
            minimap_summary = None
        if minimap_summary is not None:
            print("\n=== Minimap summary (Phase 4) ===")
            print(f"  reprojection err : {minimap_summary['reprojection_error']} m")
            print(f"  frames projected : {minimap_summary['n_frames']}")
            print(f"  field positions  : {minimap_summary['field_positions']}")
            if minimap_summary.get("minimap_video"):
                print(f"  minimap video    : {minimap_summary['minimap_video']}")
            if minimap_summary.get("overlay_video"):
                print(f"  FINAL video      : {minimap_summary['overlay_video']}")
    return 0


def _smooth_homography_with_camera_motion(provider, video_path, tracks_json, logger):
    """Propagate keypoint anchors across the video via camera motion.

    Returns a per-frame provider that follows the camera smoothly (no
    keyframe snapping), or ``None`` to fall back to the nearest-keyframe
    provider if motion estimation can't produce anything usable.
    """
    try:
        from calibration.camera_motion import (
            CameraMotionEstimator,
            build_propagated_provider,
        )

        tracks_by_frame = None
        if tracks_json:
            from role_refinement.role_refiner import load_frame_tracks
            frames, _size, _meta = load_frame_tracks(tracks_json)
            tracks_by_frame = {f.frame_index: f.tracks for f in frames}

        logger.info("Estimating camera motion to smooth the homography ...")
        transforms = CameraMotionEstimator().estimate_video(
            video_path, tracks_by_frame=tracks_by_frame)
        if not transforms:
            return None
        # Sweep every frame from the first to the last we know about.
        last = max(max(transforms), max(provider.frame_numbers))
        frame_indices = range(1, last + 1)
        smoothed = build_propagated_provider(
            provider._entries, transforms, frame_indices,
            anchor_error=provider.reprojection_error())
        if smoothed is None:
            return None
        logger.info(
            "Camera-motion smoothing: %d anchors -> %d per-frame homographies",
            len(provider), len(smoothed))
        return smoothed
    except Exception:
        logger.exception("Camera-motion smoothing failed; using keyframe homography")
        return None


def _fixed_homography_for_video(config, video_path, logger):
    """A homography fixed/stored under the video's NAME, used before auto-calib.

    Lets a video keep its own calibration so the minimap is stable for it
    regardless of the (sometimes shaky) auto-calibration. Looked up by stem:
      * ``outputs/<stem>_homography.json`` -- a hard-coded 3x3 ``{"matrix":...}``
        (one fixed homography for the whole clip) or ``{"frames":[{"frame",
        "matrix"}]}`` (per-keyframe), OR
      * ``outputs/<stem>_pitch_calibration.json`` -- a saved manual (clicked)
        calibration.
    Returns a provider (``Homography`` / ``MultiHomography``) or ``None``.
    """
    import json

    import numpy as np

    from calibration.homography import Homography, MultiHomography

    output_dir = Path(config.video.output_dir)
    stem = Path(video_path).stem

    matrix_file = output_dir / f"{stem}_homography.json"
    if matrix_file.is_file():
        try:
            data = json.loads(matrix_file.read_text(encoding="utf-8"))
            if data.get("frames"):
                entries = [(int(f["frame"]), Homography(np.array(f["matrix"], float)))
                           for f in data["frames"]]
                logger.info("Using fixed per-frame homography %s (%d keyframes)",
                            matrix_file, len(entries))
                return MultiHomography(entries)
            if "matrix" in data:
                logger.info("Using fixed homography matrix %s", matrix_file)
                return Homography(np.array(data["matrix"], float))
        except Exception:
            logger.exception("Bad fixed-homography file %s; ignoring", matrix_file)

    pts_file = output_dir / f"{stem}_pitch_calibration.json"
    if pts_file.is_file():
        from calibration.calibration_store import MultiCalibrationStore
        calibration = MultiCalibrationStore(pts_file).load()
        if calibration is not None and calibration.usable_keyframes():
            logger.info("Using saved manual calibration %s (by video name)", pts_file)
            try:
                return MultiHomography.from_keyframes(calibration.usable_keyframes())
            except ValueError:
                return None
    return None


def _build_homography(config, video_path, logger, tracks_json=None):
    """Build the homography source: auto keypoint model, or the manual file."""
    cal = config.calibration
    # A calibration fixed under the video's name wins over auto (an explicit
    # --calibration path still takes precedence over both).
    if not cal.calibration_path:
        fixed = _fixed_homography_for_video(config, video_path, logger)
        if fixed is not None:
            return fixed
    if cal.auto:
        from calibration.pitch_keypoints import KeypointHomographyProvider

        if (cal.keypoint_backend or "yolo_pose").lower() == "nbjw":
            from calibration.nbjw_keypoints import NBJWPitchDetector
            if not Path(cal.nbjw_model).is_file():
                logger.warning(
                    "NBJW backend selected but model not found at %s; "
                    "skipping minimap.", cal.nbjw_model)
                return None
            logger.info("Auto-calibrating from NBJW HRNet model %s ...", cal.nbjw_model)
            detector = NBJWPitchDetector(
                cal.nbjw_model, device=config.model.device,
                kp_threshold=cal.nbjw_kp_threshold,
                min_points=cal.keypoint_min_points,
                max_error_m=cal.keypoint_max_error_m)
        else:
            from calibration.pitch_keypoints import KeypointPitchDetector
            if not Path(cal.keypoint_model).is_file():
                logger.warning(
                    "Auto-calibration on but keypoint model not found at %s; "
                    "skipping minimap.", cal.keypoint_model)
                return None
            logger.info("Auto-calibrating from keypoint model %s ...",
                        cal.keypoint_model)
            detector = KeypointPitchDetector(
                cal.keypoint_model, device=config.model.device,
                min_conf=cal.keypoint_min_conf, min_points=cal.keypoint_min_points,
                max_error_m=cal.keypoint_max_error_m)
        provider = KeypointHomographyProvider.from_video(
            video_path, detector, sample_every=cal.keypoint_sample_every)
        if provider is None:
            logger.warning(
                "Keypoint model calibrated 0 frames; skipping minimap.")
            return None
        logger.info("Auto-calibration: %d anchor frames, mean reproj error %.2f m",
                    len(provider), provider.reprojection_error())
        if cal.camera_motion:
            smoothed = _smooth_homography_with_camera_motion(
                provider, video_path, tracks_json, logger)
            if smoothed is not None:
                return smoothed
        return provider

    # Manual calibration file (single or multi-keyframe).
    if not cal.calibration_path:
        logger.warning("No calibration (manual file or --auto-calibration); "
                       "skipping minimap.")
        return None
    from calibration.calibration_store import MultiCalibrationStore
    from calibration.homography import MultiHomography

    calibration = MultiCalibrationStore(cal.calibration_path).load()
    if calibration is None:
        logger.warning("Calibration not found at %s; skipping minimap.",
                       cal.calibration_path)
        return None
    usable = calibration.usable_keyframes()
    if not usable:
        logger.warning("Calibration has no keyframe with >=4 points; skipping.")
        return None
    try:
        return MultiHomography.from_keyframes(usable)
    except ValueError as exc:
        logger.warning("Homography failed (%s); skipping minimap.", exc)
        return None


def run_possession(config, video_path, summary, role_summary):
    """Phase 6: per-team ball possession (image-space, no calibration)."""
    from analytics.possession import (
        compute_possession,
        export_possession,
        export_possession_txt,
    )
    from calibration.minimap import load_roles_map
    from role_refinement.role_refiner import load_frame_tracks

    tracks_json = summary.get("tracks_json")
    frames, _frame_size, _meta = load_frame_tracks(tracks_json)
    roles_path = _resolve_roles_path(config, video_path, role_summary)
    if not roles_path:
        get_logger("main.possession").warning(
            "No roles JSON for possession; skipping.")
        return None
    roles_map = load_roles_map(roles_path)

    pc = config.possession
    result = compute_possession(
        frames, roles_map,
        ball_class_name=config.tracking.ball_class_name,
        gate_factor=pc.gate_factor, min_hold=pc.min_hold)

    output_dir = Path(config.video.output_dir)
    stem = Path(video_path).stem
    json_path = export_possession(
        result, output_dir / f"{stem}_possession.json",
        metadata={"source_tracks": str(tracks_json), "roles_source": roles_path})
    export_possession_txt(
        result, output_dir / f"{stem}_possession.txt", title=stem)

    # Hand the running cumulative bar data to the final-video renderer.
    if pc.show_bar:
        summary["possession_cum"] = {
            frame: (p0, p1) for frame, p0, p1 in result.cumulative}
    return {"pct": result.percentages, "json": str(json_path)}


def run_minimap(config, video_path, summary, role_summary):
    """Phase 4: project tracks to the pitch and render the minimap."""
    logger = get_logger("main.minimap")
    from calibration import minimap as mm
    from role_refinement.role_refiner import load_frame_tracks

    homography = _build_homography(
        config, video_path, logger, tracks_json=summary.get("tracks_json"))
    if homography is None:
        return None
    cal_path = (
        "keypoint-model:auto" if config.calibration.auto
        else config.calibration.calibration_path)

    error = homography.reprojection_error()
    if error > config.calibration.reprojection_warn_threshold:
        logger.warning(
            "High reprojection error: %.2f m (threshold %.1f m) — minimap "
            "positions may be inaccurate.",
            error, config.calibration.reprojection_warn_threshold)

    tracks_json = summary.get("tracks_json")
    if not tracks_json:
        logger.warning("No tracks JSON available; skipping minimap.")
        return None

    frames, _frame_size, meta = load_frame_tracks(tracks_json)
    roles_path = _resolve_roles_path(config, video_path, role_summary)
    roles_map = mm.load_roles_map(roles_path) if roles_path else {}
    names_map = mm.load_player_names(roles_path) if roles_path else {}

    renderer = mm.MinimapRenderer(
        px_per_meter=config.minimap.px_per_meter,
        player_radius=config.minimap.player_dot_radius,
        ball_radius=config.minimap.ball_dot_radius,
        stripes=config.minimap.stripes,
        hide_referees=config.minimap.hide_referees,
        tactical_style=config.minimap.tactical_style,
    )
    color_mode = config.minimap.color_mode
    show_ids = config.minimap.show_track_ids

    # Colour each player by the shirt they wear in each frame (not the track's
    # team label), so an ID switch can't paint a player the wrong team.
    team_override = None
    if config.minimap.team_color_per_frame and color_mode == mm.COLOR_MODE_TEAM_ROLE:
        team_override = mm.classify_per_frame_teams(
            frames, video_path, roles_map,
            ball_class_name=config.tracking.ball_class_name)
        if team_override:
            logger.info("Per-frame team colouring on: %d player-frames classified",
                        len(team_override))

    positions = mm.project_tracks_to_field(
        frames, roles_map, homography, renderer, color_mode=color_mode,
        smooth_alpha=config.minimap.smooth_alpha,
        max_step_m=config.minimap.max_step_m,
        team_override=team_override,
        names_map=names_map)
    logger.info("Minimap coloring: %s (show_track_ids=%s)", color_mode, show_ids)

    output_dir = Path(config.video.output_dir)
    stem = Path(video_path).stem
    fp_path = config.calibration.field_positions_output or str(
        output_dir / f"{stem}_field_positions.json")
    mm.write_field_positions(positions, fp_path, metadata={
        "phase": "calibration_minimap",
        "source_tracks": str(tracks_json),
        "roles_source": roles_path,
        "calibration": str(cal_path),
        "reprojection_error_m": round(float(error), 4),
    })

    fps = float(meta.get("video_fps", 25.0))

    # Phase 5: speed & distance from the field metres we just projected.
    # Never crash the pipeline — analytics is optional post-processing.
    speed_lookup = None
    ac = config.analytics
    if ac.enabled and positions:
        try:
            from analytics import (
                build_stat_lookup,
                compute_analytics,
                export_analytics,
                export_analytics_txt,
            )

            result = compute_analytics(
                positions, roles_map, fps,
                smoothing_window=ac.smoothing_window, max_speed_kmh=ac.max_speed_kmh,
                min_samples=ac.min_samples,
                ball_class_name=config.tracking.ball_class_name,
                exclude_roles=ac.exclude_roles,
                min_track_samples=ac.min_track_samples,
                player_names=names_map)
            analytics_path = export_analytics(
                result, output_dir / f"{stem}_analytics.json",
                metadata={"source_field_positions": fp_path})
            report_path = export_analytics_txt(
                result, output_dir / f"{stem}_analytics.txt", title=stem)
            print("\n=== Speed & distance (Phase 5) ===")
            print(f"  tracks analysed  : {len(result.tracks)}")
            print(f"  total distance   : {result.total_distance_m:.0f} m")
            for team in (0, 1):
                print(f"  team {team} distance : {result.team_distance(team):.0f} m")
            print(f"  analytics JSON   : {analytics_path}")
            print(f"  analytics report : {report_path}")

            # Phase 6: per-player performance (rating / work-rate / insight).
            profiles = []          # Phase 8 fills these; kept for Phase 10.
            from analytics import (
                compute_player_analytics,
                export_player_analytics,
                export_player_analytics_txt,
            )
            players = compute_player_analytics(result)
            if players:
                pa_path = export_player_analytics(
                    players, output_dir / f"{stem}_player_analytics.json",
                    metadata={"source_field_positions": fp_path})
                pa_txt = export_player_analytics_txt(
                    players, output_dir / f"{stem}_player_analytics.txt", title=stem)
                top = players[0]
                print("\n=== Player performance (Phase 6) ===")
                print(f"  players rated    : {len(players)}")
                top_label = (
                    f"{top.player_name} (#{top.track_id})"
                    if top.player_name else top.player_display_name)
                print(f"  top rated        : {top_label} "
                      f"({top.player_rating}/10, {top.performance_status})")
                print(f"  player analytics : {pa_path}")
                print(f"  player report    : {pa_txt}")

            # Phase 7: team + best-player heatmaps (PNG).
            if ac.heatmaps:
                from analytics import collect_positions, density_grid, save_heatmap
                heatmaps = []
                for team in (0, 1):
                    tpts = collect_positions(positions, team_id=team)
                    if tpts:
                        save_heatmap(
                            density_grid(tpts, sigma_m=ac.heatmap_sigma_m),
                            output_dir / f"{stem}_team{team}_heatmap.png",
                            label=f"Team {team}")
                        heatmaps.append(team)
                    team_players = [
                        t for t in result.tracks
                        if t.role == "player" and t.team_id == team]
                    if team_players:
                        best = max(team_players, key=lambda t: t.total_distance_m)
                        ppts = collect_positions(positions, track_id=best.track_id)
                        if ppts:
                            label = (
                                f"Team {team} {best.player_name}"
                                if best.player_name
                                else f"Team {team} {best.player_display_name}")
                            save_heatmap(
                                density_grid(ppts, sigma_m=ac.heatmap_sigma_m),
                                output_dir
                                / f"{stem}_team{team}_best_player_heatmap.png",
                                label=label)
                if heatmaps:
                    print("\n=== Heatmaps (Phase 7) ===")
                    print(f"  written for teams: {heatmaps} "
                          f"(+ best player each) -> {output_dir}/{stem}_team*.png")

            # Phase 8: team tactical analysis (explainable labels + reasons).
            if ac.tactical:
                from analytics import (
                    compute_tactical_analysis,
                    export_team_tactical,
                    export_team_tactical_txt,
                )
                profiles = compute_tactical_analysis(
                    positions, result,
                    ball_class_name=config.tracking.ball_class_name)
                if profiles:
                    tac_path = export_team_tactical(
                        profiles, output_dir / f"{stem}_team_tactical.json",
                        metadata={"source_field_positions": fp_path})
                    tac_txt = export_team_tactical_txt(
                        profiles, output_dir / f"{stem}_team_tactical.txt", title=stem)
                    print("\n=== Team tactical analysis (Phase 8) ===")
                    for p in profiles:
                        print(f"  team {p.team_id}: {p.compactness} / "
                              f"{p.pressure_style} / attacks {p.attacking_zone} / "
                              f"{p.build_up_style}")
                    print(f"  tactical JSON    : {tac_path}")
                    print(f"  tactical report  : {tac_txt}")

            # Phase 9: field-space ball possession (projected metres). Writes
            # the authoritative <stem>_possession.json/.txt (more accurate than
            # the image-space estimate, which still drives the on-video bar).
            from analytics import (
                compute_field_possession,
                export_field_possession,
                export_field_possession_txt,
            )
            poss = compute_field_possession(
                positions, ball_class_name=config.tracking.ball_class_name)
            if poss.possessed_frames:
                export_field_possession(
                    poss, output_dir / f"{stem}_possession.json",
                    metadata={"source_field_positions": fp_path})
                export_field_possession_txt(
                    poss, output_dir / f"{stem}_possession.txt", title=stem)
                print("\n=== Ball possession — field (Phase 9) ===")
                print(f"  team 0 / team 1  : {poss.percentages[0]}% / "
                      f"{poss.percentages[1]}%")
                print(f"  dominant team    : {poss.dominant_team}")

            # Phase 10: AI match intelligence — the final analyst report.
            if ac.final_report and players:
                from analytics import (
                    build_match_report,
                    export_final_report,
                    export_final_report_txt,
                )
                report = build_match_report(players, profiles, poss)
                fr_json = export_final_report(
                    report, output_dir / f"{stem}_final_report.json",
                    metadata={"source_field_positions": fp_path})
                fr_txt = export_final_report_txt(
                    report, output_dir / f"{stem}_final_report.txt", title=stem)
                print("\n=== Match report (Phase 10) ===")
                print(f"  dominant team    : {report.dominant_team}")
                motm = (
                    f"{report.man_of_the_match_name} (#{report.man_of_the_match})"
                    if report.man_of_the_match_name
                    else report.man_of_the_match_display_name)
                print(f"  man of the match : {motm}")
                print(f"  summary          : {report.summary}")
                print(f"  final report     : {fr_json}")
                print(f"  final report txt : {fr_txt}")

            if ac.show_speed_overlay:
                speed_lookup = build_stat_lookup(result)
        except Exception:
            logger.exception("Speed/distance analytics failed (continuing)")
    elif ac.enabled:
        logger.warning("No field positions available; skipping speed/distance.")

    video_out = None
    if config.calibration.save_minimap:
        mm_path = config.calibration.minimap_output or str(
            output_dir / f"{stem}_minimap.mp4")
        video_out = str(mm.render_minimap_video(
            positions, renderer, mm_path, fps=fps, show_track_ids=show_ids))

    # Combined final video the user sees: team/role-colored player boxes +
    # role labels (corrections applied) + the minimap as picture-in-picture,
    # all rendered from the original source video in one pass.
    overlay_out = None
    overlay_cfg = config.minimap_overlay
    pose_estimator = None
    if config.pose.enabled:
        from visualization.pose_overlay import PoseEstimator
        pose_estimator = PoseEstimator(
            config.pose.model, device=config.model.device, conf=config.pose.conf,
            ball_class_name=config.tracking.ball_class_name)
    if overlay_cfg.enabled:
        overlay_path = output_dir / f"{stem}_final_with_minimap.mp4"
        overlay_out = str(mm.render_final_video(
            video_path, frames, positions, roles_map, renderer, overlay_path,
            color_mode=color_mode,
            show_track_ids=show_ids,
            position=overlay_cfg.position,
            width_ratio=overlay_cfg.width_ratio,
            margin=overlay_cfg.margin,
            background_alpha=overlay_cfg.background_alpha,
            border=overlay_cfg.border,
            fps=fps,
            possession_cum=summary.get("possession_cum"),
            pose_estimator=pose_estimator,
            pose_min_conf=config.pose.min_keypoint_conf,
            pose_box_fallback=config.pose.draw_box_fallback,
            team_override=team_override,
            speed_lookup=speed_lookup,
            names_map=names_map))

    return {
        "field_positions": fp_path,
        "minimap_video": video_out,
        "overlay_video": overlay_out,
        "reprojection_error": round(float(error), 3),
        "n_frames": len(positions),
    }


def _resolve_roles_path(config, video_path, role_summary):
    """Prefer the final (corrected) roles, then auto roles, else None."""
    output_dir = Path(config.video.output_dir)
    stem = Path(video_path).stem
    final = Path(
        config.manual_correction.final_output_path
        or output_dir / f"{stem}_roles_final.json"
    )
    if final.is_file():
        return str(final)
    if role_summary and role_summary.get("roles_json"):
        return role_summary["roles_json"]
    auto = output_dir / f"{stem}_roles.json"
    return str(auto) if auto.is_file() else None


def _find_reusable_tracks(config, video_path):
    """Locate the most-processed tracks JSON from a previous run, or None.

    Prefers the swap-corrected file, then the stitched, then the raw tracks
    (the same precedence the live pipeline produces), so a ``--reuse-tracks``
    run picks up exactly the identities a prior ``--all`` ended with.
    """
    output_dir = Path(config.video.output_dir)
    stem = Path(video_path).stem
    for name in (f"{stem}_tracks_manualswap.json",
                 f"{stem}_tracks_swapfixed.json",
                 f"{stem}_tracks_stitched.json",
                 f"{stem}_tracks.json"):
        candidate = output_dir / name
        if candidate.is_file():
            return str(candidate)
    return None


def _fresh_against_tracks(human_file, tracks_json) -> bool:
    """True if a human-edited file was made on the *current* tracks.

    Init labels and role corrections are both keyed to track ids, which a
    fresh detection pass renumbers, so a file from an earlier run points at
    the wrong people. We accept it only when it is at least as new as the
    tracks file it would be applied to (i.e. it was saved *after* these
    tracks were produced). Missing files are never "fresh".
    """
    human_path = Path(human_file)
    if not human_path.is_file() or not tracks_json:
        return False
    tracks_path = Path(tracks_json)
    if not tracks_path.is_file():
        return False
    return human_path.stat().st_mtime >= tracks_path.stat().st_mtime


def run_manual_correction(config, args, video_path, summary, role_summary):
    """Phase 3.5: build the review dataset and/or apply corrections."""
    from manual_correction.correction_runner import RoleCorrectionRunner

    mc = config.manual_correction
    runner = RoleCorrectionRunner(mc)
    roles_json = role_summary["roles_json"]
    tracks_json = summary.get("tracks_json")
    output_dir = Path(config.video.output_dir)
    stem = Path(video_path).stem

    if not tracks_json:
        print("  No tracks JSON available; skipping manual correction.")
        return None

    dataset = runner.build_review_dataset(video_path, tracks_json, roles_json)
    print("\n=== Manual review + names (Phase 3.5) ===")
    print(f"  review tracks    : {dataset['n_candidates']}")
    print(f"  review manifest  : {dataset['manifest']}")
    if dataset["n_candidates"]:
        from manual_correction.correction_tk_ui import (
            CorrectionApp,
            CorrectionSession,
        )

        print("  launching Tkinter review UI "
              "(save edits, then Finish Review to continue)...")
        session = CorrectionSession(
            dataset["manifest"], corrections_path=mc.corrections_file)
        confirmed = CorrectionApp(session).run()
        if not confirmed:
            raise RuntimeError(
                "Manual correction review was closed without confirmation; "
                "analytics/results were not generated."
            )
    else:
        print("  No non-ball tracks to review.")

    # Auto-merge saved initialization labels (multi-select team seeds) so
    # the reviewer's manual teams are never silently dropped -- BUT only
    # when those labels were made on THESE tracks.
    if _fresh_against_tracks(mc.initialization_labels_file, tracks_json):
        seeds = runner.apply_initialization(
            mc.initialization_labels_file, mc.corrections_file, overwrite=False)
        print(f"  applied {seeds['n_seeds']} initialization seed(s) "
              f"from {mc.initialization_labels_file}")
    elif Path(mc.initialization_labels_file).is_file():
        print(f"  skipped stale initialization labels "
              f"({mc.initialization_labels_file} predates the current "
              f"tracks; re-label to apply, or use --reuse-tracks)")

    # Corrections are track-id keyed; do not apply files from older tracks.
    corrections_arg = mc.corrections_file
    if not _fresh_against_tracks(mc.corrections_file, tracks_json):
        if Path(mc.corrections_file).is_file():
            print(f"  skipped stale corrections ({mc.corrections_file} "
                  f"predates the current tracks; using automatic roles)")
        corrections_arg = str(output_dir / f"{stem}_no_corrections.json")
    final_output = mc.final_output_path or str(
        output_dir / f"{stem}_roles_final.json"
    )
    result = runner.run_apply(
        video_path=video_path,
        roles_json_path=roles_json,
        corrections_file=corrections_arg,
        final_output_path=final_output,
        tracks_json_path=tracks_json,
        render_video=mc.save_final_video,
    )
    print("\n=== Final roles summary (Phase 3.5) ===")
    for name, count in sorted(result["final_by_role"].items()):
        print(f"    {name:<12}: {count}")
    print(f"  tracks           : {result['n_tracks']}")
    print(f"  user-corrected   : {result['n_corrected']}")
    print(f"  player names     : {result['n_named']}")
    print(f"  final roles JSON : {result['final_roles_json']}")
    if result.get("final_video"):
        print(f"  final role video : {result['final_video']}")
    return result


def _is_interactive() -> bool:
    """True only when a real terminal is attached (so prompts can't hang)."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def _ask(prompt: str) -> str:
    """input() that returns '' on EOF/interrupt (non-interactive stdin)."""
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def ensure_calibration(config, video_path, logger) -> None:
    """If no usable pitch calibration exists, offer to launch the UI now."""
    from calibration.calibration_store import MultiCalibrationStore

    path = config.calibration.calibration_path
    existing = MultiCalibrationStore(path).load()
    if existing is not None and existing.usable_keyframes():
        return  # already calibrated

    if not _is_interactive():
        logger.warning(
            "No usable calibration at %s (and no terminal to prompt); "
            "the minimap will be skipped.", path)
        return

    print(f"\n[setup] No pitch calibration found at {path}.")
    answer = _ask(
        "        Calibrate the pitch now? (click >=4 reference points, "
        "Ctrl+S to save) [Y/n] ")
    if answer in ("n", "no"):
        print("        Skipping calibration -> the minimap will be skipped.")
        return

    from calibration.calibration_ui import main as calibration_main
    calibration_main(["--video", str(video_path), "--calibration", path])
    refreshed = MultiCalibrationStore(path).load()
    if refreshed is None or not refreshed.usable_keyframes():
        print("        Still no usable calibration -> the minimap will be skipped.")


def offer_correction_review(config, video_path, summary, role_summary, logger) -> None:
    """If no corrections exist, offer to review the ambiguous tracks now."""
    mc = config.manual_correction
    if Path(mc.corrections_file).is_file():
        return  # corrections already exist -> they'll be applied as usual
    if not _is_interactive():
        return
    roles_json = role_summary.get("roles_json")
    tracks_json = summary.get("tracks_json")
    if not roles_json or not tracks_json:
        return

    from manual_correction.correction_runner import (
        find_candidate_ids,
        load_auto_roles,
    )

    candidates = find_candidate_ids(load_auto_roles(roles_json), mc)
    if not candidates:
        return

    print(f"\n[setup] {len(candidates)} ambiguous track(s) "
          "(unknown / low confidence) could be reviewed manually.")
    answer = _ask("        Review them now? [y/N] ")
    if answer not in ("y", "yes"):
        print("        Skipping review -> using the automatic roles.")
        return

    from manual_correction.correction_runner import RoleCorrectionRunner
    from manual_correction.correction_tk_ui import (
        CorrectionApp,
        CorrectionSession,
    )

    runner = RoleCorrectionRunner(mc)
    dataset = runner.build_review_dataset(video_path, tracks_json, roles_json)
    print(f"        Launching the review UI for {dataset['n_candidates']} "
          "track(s)... (Save & Next, then Finish Review when done)")
    session = CorrectionSession(dataset["manifest"],
                                corrections_path=mc.corrections_file)
    if not CorrectionApp(session).run():
        print("        Review closed without confirmation; keeping automatic roles.")


def _sample_track_appearance(frames, video_path, config):
    """Mean jersey-colour vector per (non-ball) track, for the stitch gate.

    Reuses the role-refinement appearance pipeline (jersey-region histogram)
    so the colour space matches team clustering. Returns ``{track_id: vec}``.
    """
    import numpy as np

    from role_refinement.appearance_extractor import (
        AppearanceExtractor,
        build_track_history,
    )
    from role_refinement.role_refiner import VideoCropProvider, _invert_plan
    from utils.config_loader import RoleRefinementConfig

    rr = RoleRefinementConfig(
        samples_per_track=20, ball_class_name=config.tracking.ball_class_name)
    extractor = AppearanceExtractor(rr)
    history = build_track_history(frames)
    plan = extractor.sample_frames(history)
    provider = VideoCropProvider.from_video(
        video_path, _invert_plan(history, plan, rr))

    appearance: dict = {}
    for track_id, observations in history.items():
        if observations and observations[0].class_name == rr.ball_class_name:
            continue
        bbox_by_frame = {o.frame_id: o.bbox for o in observations}
        vecs = []
        for frame_id in plan.get(track_id, []):
            vec = extractor.vector_from_crop(
                provider.get(track_id, frame_id, bbox_by_frame.get(frame_id)))
            if vec is not None:
                vecs.append(vec)
        if vecs:
            appearance[track_id] = np.mean(np.stack(vecs, axis=0), axis=0)
    return appearance


def run_stitching(config, video_path, summary):
    """Re-link broken track IDs and repoint downstream at the stitched JSON."""
    logger = get_logger("main.stitching")
    tracks_json = summary.get("tracks_json")
    if not tracks_json:
        return None

    from role_refinement.role_refiner import load_frame_tracks
    from tracking.track_exporter import TrackJSONExporter
    from tracking.track_stitcher import stitch_tracks

    frames, _frame_size, meta = load_frame_tracks(tracks_json)
    st = config.tracking.stitching

    # Sample each track's jersey colour so two different-team players are
    # never stitched into one (switching) id -- a safety filter on the merges.
    appearance_of = None
    if st.appearance_gate:
        try:
            appearance_of = _sample_track_appearance(frames, video_path, config)
            logger.info("Appearance gate on: %d track colours sampled",
                        len(appearance_of))
        except Exception:
            logger.exception("Appearance sampling failed; stitching on geometry only")
            appearance_of = None

    result = stitch_tracks(
        frames,
        max_frame_gap=st.max_frame_gap,
        distance_gate_px=st.distance_gate_px,
        size_ratio_gate=st.size_ratio_gate,
        match_same_class=st.match_same_class,
        ball_class_name=config.tracking.ball_class_name,
        min_segment_length=st.min_segment_length,
        appearance_of=appearance_of,
        appearance_min_sim=st.appearance_min_sim,
    )

    stem = Path(video_path).stem
    out_path = Path(config.video.output_dir) / f"{stem}_tracks_stitched.json"
    stitched_meta = dict(meta)
    stitched_meta.update({
        "stitched": True,
        "ids_before": result.ids_before,
        "ids_after": result.ids_after,
        "merges": result.merges,
    })
    exporter = TrackJSONExporter(out_path, metadata=stitched_meta)
    for frame in result.frames:
        exporter.add_frame(frame.frame_index, frame.tracks)
    exporter.save()

    # Downstream (role refinement, corrections, minimap) now use the
    # stitched identities.
    summary["tracks_json"] = str(out_path)
    summary["stitching"] = {
        "ids_before": result.ids_before,
        "ids_after": result.ids_after,
        "merges": result.merges,
    }
    logger.info("Stitched tracks written to %s", out_path)
    return summary["stitching"]


def run_swap_correction(config, video_path, summary):
    """Fix crossing ID-swaps and repoint downstream at the corrected JSON."""
    logger = get_logger("main.swap_correction")
    tracks_json = summary.get("tracks_json")
    if not tracks_json:
        return None

    from role_refinement.role_refiner import load_frame_tracks
    from tracking.swap_corrector import SwapCorrector
    from tracking.track_exporter import TrackJSONExporter

    frames, _frame_size, meta = load_frame_tracks(tracks_json)
    sc = config.tracking.swap_correction
    corrector = SwapCorrector(
        ball_class_name=config.tracking.ball_class_name,
        samples_per_track=sc.samples_per_track,
        min_track_length=sc.min_track_length,
        frame_tol=sc.frame_tol,
        distance_gate_px=sc.distance_gate_px,
        min_purity=sc.min_purity,
        split_unpaired=sc.split_unpaired,
        min_split_samples=sc.min_split_samples,
        min_split_purity=sc.min_split_purity,
    )
    result = corrector.correct(frames, video_path)
    if result.n_swaps == 0 and not result.splits:
        return {"flips": "-", "swaps": 0, "splits": 0, "pairs": []}

    stem = Path(video_path).stem
    out_path = Path(config.video.output_dir) / f"{stem}_tracks_swapfixed.json"
    fixed_meta = dict(meta)
    fixed_meta.update({"swap_corrected": True, "swaps": result.n_swaps,
                       "drift_splits": len(result.splits)})
    exporter = TrackJSONExporter(out_path, metadata=fixed_meta)
    for frame in result.frames:
        exporter.add_frame(frame.frame_index, frame.tracks)
    exporter.save()

    summary["tracks_json"] = str(out_path)
    logger.info("Swap-corrected tracks written to %s", out_path)
    return {
        "flips": "-",
        "swaps": result.n_swaps,
        "splits": len(result.splits),
        "pairs": [f"#{a}<->#{b}@{s}-{e}" for a, b, s, e in result.swaps],
    }


def run_roster(config, video_path, summary):
    """Replace free-running track ids with a fixed squad: 1..11, 12..22, 0.

    Runs after every id-repairing step and before role refinement, so the rest
    of the pipeline sees squad numbers rather than whatever the tracker
    happened to mint. That ordering is the point: with a bounded set of ids, a
    missing player shows up as an empty shirt instead of hiding among a surplus
    of ids nobody counted.

    Who is in a squad is decided twice over, because neither signal is enough
    alone. Colour puts the two kits into two clusters and everything else into
    a third -- but a goalkeeper is *required* to wear something the outfield
    players do not, so he lands in that third group beside the officials, and a
    colour-only rule would strike him off. Geometry then separates the third
    group by what its members do on the ground: an assistant hugs a touchline,
    a keeper holds the area in front of one goal, the referee roams the middle.
    The keeper goes back into his squad and the officials stay out.

    Geometry needs a calibration. Without one this falls back to colour alone
    and says so, because a roster that occasionally benches a goalkeeper is
    still better than 27 ids for 22 people -- but it is worth knowing which of
    the two ran.
    """
    logger = get_logger("main.roster")
    tracks_json = summary.get("tracks_json")
    if not tracks_json:
        return None

    import cv2
    import numpy as np

    from analytics.pitch_role import (
        belongs_in_a_squad,
        best_deepest_share,
        classify,
    )
    from role_refinement.appearance_extractor import (
        AppearanceExtractor,
        build_track_history,
    )
    from role_refinement.role_refiner import (
        VideoCropProvider,
        _invert_plan,
        load_frame_tracks,
    )
    from tracking.roster import apply
    from tracking.track_exporter import TrackJSONExporter
    from utils.config_loader import RoleRefinementConfig

    cfg = config.tracking.roster
    frames, _frame_size, meta = load_frame_tracks(tracks_json)
    if not frames:
        return None

    rr = RoleRefinementConfig(ball_class_name=config.tracking.ball_class_name)
    extractor = AppearanceExtractor(rr)
    history = build_track_history(frames)
    plan = extractor.sample_frames(history)
    provider = VideoCropProvider.from_video(
        video_path, _invert_plan(history, plan, rr))

    shirts, sampled = {}, {}
    for track_id, observations in history.items():
        if observations and observations[0].class_name == rr.ball_class_name:
            continue
        boxes = {o.frame_id: o.bbox for o in observations}
        colours, frames_used = [], []
        for frame_id in plan.get(track_id, []):
            colour = extractor.median_bgr(
                provider.get(track_id, frame_id, boxes.get(frame_id)))
            if colour is not None:
                colours.append(colour)
                frames_used.append(frame_id)
        if colours:
            shirts[track_id] = np.median(np.stack(colours, axis=0), axis=0)
            sampled[track_id] = frames_used

    if len(shirts) < 3:
        logger.info("Roster: too few tracks to tell the kits apart; skipping")
        return None

    # Three shirt groups, not two: the officials and the keepers wear neither
    # kit, and forcing them into the nearer one is what puts a referee in a
    # team. The two most populous groups are the teams.
    #
    # Raw medians rather than the appearance histogram. The histogram is
    # normalised for comparing two shirts and that normalisation blurs how
    # FAR apart three kinds of shirt are: clustered on histograms the referee
    # joined a team and wore squad number 22, while on medians his yellow is
    # 160 apart from either kit and lands in its own group.
    ids = sorted(shirts)
    data = np.float32([shirts[i] for i in ids])
    _compact, labels, _centres = cv2.kmeans(
        data, 3, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.1),
        16, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel().tolist()
    counts = Counter(labels)
    kits = [group for group, _ in counts.most_common(2)]

    homography = _build_homography(config, video_path, logger, tracks_json)
    boxes_at = {tid: {o.frame_id: o.bbox for o in obs}
                for tid, obs in history.items()}

    # Pitch positions on a regular frame grid rather than at each track's own
    # sample frames: telling who is deepest means comparing tracks against each
    # other, and the per-track sampling plan gives them almost no frames in
    # common to be compared on.
    picker = getattr(homography, "for_frame", None) if homography else None
    walked = defaultdict(dict)                 # track -> {frame: (x, y) metres}
    if homography is not None:
        for track_id, observations in history.items():
            for o in observations:
                if o.frame_id % cfg.pitch_sample_every:
                    continue
                H = picker(o.frame_id) if picker else homography
                x, y = H.project_image_to_field(
                    ((o.bbox[0] + o.bbox[2]) / 2.0, o.bbox[3]))
                if math.isfinite(x) and math.isfinite(y):
                    walked[track_id][o.frame_id] = (float(x), float(y))

    # Where each team stood, frame by frame, so "behind all of them" can be
    # asked of anyone the kits left out.
    team_positions = [defaultdict(list), defaultdict(list)]
    for index, track_id in enumerate(ids):
        if labels[index] not in kits:
            continue
        side = kits.index(labels[index])
        for frame, point in walked.get(track_id, {}).items():
            team_positions[side][frame].append(point)

    team_of, roles = {}, Counter()
    for index, track_id in enumerate(ids):
        wears_kit = labels[index] in kits
        track = walked.get(track_id, {})
        # Only worth computing for someone outside both kits: a keeper in a
        # team shirt is an outfield player as far as numbering is concerned.
        behind = 0.0 if wears_kit else best_deepest_share(track, team_positions)
        role = classify(list(track.values()), wears_kit, behind_team=behind)
        roles[role.value] += 1
        if belongs_in_a_squad(role):
            side = kits.index(labels[index]) if wears_kit else _nearest_team(
                shirts[track_id], data, labels, kits)
            team_of[track_id] = side

    result = apply(
        frames, team_of,
        slots_per_team=cfg.slots_per_team,
        max_gap_frames=cfg.max_gap_frames,
        distance_gate_px=cfg.distance_gate_px,
        ball_class_name=config.tracking.ball_class_name,
        unnumbered_classes=tuple(cfg.unnumbered_classes),
    )

    stem = Path(video_path).stem
    out_path = Path(config.video.output_dir) / f"{stem}_tracks_roster.json"
    roster_meta = dict(meta)
    roster_meta.update({"roster": True, "slots_used": result.slots_used,
                        "overflow": len(result.overflow),
                        "pitch_roles": bool(homography)})
    exporter = TrackJSONExporter(out_path, metadata=roster_meta)
    for frame in result.frames:
        exporter.add_frame(frame.frame_index, frame.tracks)
    exporter.save()

    summary["tracks_json"] = str(out_path)
    logger.info("Roster written to %s", out_path)
    return {
        "numbered": len(result.assignment),
        "slots_used": result.slots_used,
        "overflow": len(result.overflow),
        "roles": dict(roles),
        "by_position": homography is not None,
    }


def _nearest_team(vector, data, labels, kits):
    """Which kit a non-kit shirt sits closest to -- used only for a keeper.

    A goalkeeper has already been identified by where he plays; this only has
    to decide which half of the squad numbers he takes, and the nearer kit is
    the best available guess. It can be wrong, and when it is, the keeper wears
    the other side's number -- visible, and fixable by hand.
    """
    import numpy as np

    best, side = None, 0
    for group in kits:
        members = np.stack([data[i] for i, g in enumerate(labels) if g == group])
        distance = float(np.linalg.norm(vector - members.mean(axis=0)))
        if best is None or distance < best:
            best, side = distance, kits.index(group)
    return side


def run_manual_swap(config, video_path, summary):
    """Apply reviewer-marked ID-switch swaps to the tracks (before roles).

    Reads the manual corrections file; for tracks flagged with ``id_switch``
    + ``merge_with_track_id`` + ``switch_frame`` it exchanges the two ids from
    the switch frame to the end, so each player keeps one id for the whole
    clip. Always re-derives from the (stable) auto swap-corrected tracks, so
    re-running is idempotent. Repoints downstream at the corrected JSON.
    """
    logger = get_logger("main.manual_swap")
    from manual_correction.correction_store import CorrectionStore
    from manual_correction.manual_swap import (
        build_manual_swaps,
        resolve_swap_source,
    )
    from role_refinement.role_refiner import load_frame_tracks
    from tracking.swap_corrector import apply_swaps
    from tracking.track_exporter import TrackJSONExporter

    corrections_file = config.manual_correction.corrections_file
    if not Path(corrections_file).is_file():
        return None
    output_dir = Path(config.video.output_dir)
    stem = Path(video_path).stem
    source = resolve_swap_source(output_dir, stem)
    if source is None:
        return None
    # Corrections are track-id keyed: only apply ones made on THESE tracks.
    if not _fresh_against_tracks(corrections_file, source):
        return None

    corrections = CorrectionStore(corrections_file).load()
    frames, _frame_size, meta = load_frame_tracks(str(source))
    if not frames:
        return None
    last_frame = max(f.frame_index for f in frames)
    swaps = build_manual_swaps(corrections, last_frame)
    if not swaps:
        return None

    corrected = apply_swaps(frames, swaps)
    out_path = output_dir / f"{stem}_tracks_manualswap.json"
    fixed_meta = dict(meta)
    fixed_meta.update({"manual_swap": True, "manual_swaps": len(swaps)})
    exporter = TrackJSONExporter(out_path, metadata=fixed_meta)
    for frame in corrected:
        exporter.add_frame(frame.frame_index, frame.tracks)
    exporter.save()

    summary["tracks_json"] = str(out_path)
    # Re-save the (unchanged) corrections so they stay newer than the tracks
    # file we just wrote -- keeps the downstream apply step's freshness gate
    # satisfied without forcing the user to re-label.
    CorrectionStore(corrections_file).save(corrections)
    logger.info("Manual swap-corrected tracks written to %s", out_path)
    return {
        "swaps": len(swaps),
        "pairs": [f"#{a}<->#{b}@{s}-{e}" for a, b, s, e in swaps],
    }


def run_role_refinement(config, video_path, summary):
    """Run Phase 3 after the detection/tracking pipeline has finished."""
    logger = get_logger("main.role_refinement")
    tracks_json = summary.get("tracks_json")
    if not tracks_json:
        logger.error(
            "Role refinement needs a tracks JSON, but tracking produced none."
        )
        return None

    from role_refinement.role_refiner import RoleRefinementRunner

    output_dir = Path(config.video.output_dir)
    stem = Path(video_path).stem
    roles_output = config.role_refinement.output_path or str(
        output_dir / f"{stem}_roles.json"
    )
    role_video = (
        str(output_dir / f"{stem}_roles.mp4")
        if config.role_refinement.save_role_video
        else None
    )

    runner = RoleRefinementRunner(config.role_refinement)
    return runner.run(
        video_path=video_path,
        tracks_json_path=tracks_json,
        roles_output_path=roles_output,
        role_video_path=role_video,
    )


if __name__ == "__main__":
    sys.exit(main())
