"""Typed configuration loading.

The YAML config file is parsed into dataclasses so that the rest of the
codebase works with validated, typed objects instead of raw dictionaries.
Unknown keys are ignored; missing keys fall back to sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    weights_path: str = "weights/best.pt"
    device: str = "auto"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.50
    image_size: int = 1280
    class_confidence_overrides: Dict[str, float] = field(default_factory=dict)


@dataclass
class VideoConfig:
    output_dir: str = "outputs"
    save_video: bool = True
    output_codec: str = "mp4v"


@dataclass
class VisualizationConfig:
    box_thickness: int = 2
    font_scale: float = 0.5
    show_confidence: bool = True
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)


@dataclass
class DebugConfig:
    enabled: bool = False
    show_window: bool = False
    save_debug_frames: bool = False
    debug_frame_interval: int = 50


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "outputs/logs"
    log_to_file: bool = True


@dataclass
class BotSortParams:
    """Per-tracker BoT-SORT association parameters.

    Mirrors Ultralytics' ``botsort.yaml`` so the values can be passed
    straight through to the underlying ``BOTSORT`` tracker.
    """

    track_high_thresh: float = 0.25   # first-stage match threshold
    track_low_thresh: float = 0.1     # second-stage (low-score) threshold
    new_track_thresh: float = 0.25    # min score to start a brand-new track
    track_buffer: int = 30            # frames a lost track is kept alive
    match_thresh: float = 0.8         # association cost threshold
    fuse_score: bool = True           # fuse detection score into IoU cost
    gmc_method: str = "sparseOptFlow"  # global motion compensation method
    proximity_thresh: float = 0.5     # IoU gate for ReID (when enabled)
    appearance_thresh: float = 0.8    # appearance gate for ReID (when enabled)


@dataclass
class BallTrackingConfig:
    """Ball-specific association (custom BallTracker, no Ultralytics).

    The ball is small (~6-7 px) and fast (~8-9 px/frame), so consecutive
    detections often have zero IoU and BoT-SORT cannot link them. This
    tracker inflates the ball box *for matching only*, and falls back to
    center-distance matching (with velocity prediction) when IoU is zero.
    """

    enabled: bool = True
    matching_box_scale: float = 3.0   # inflate ball box by this factor to match
    min_box_size: float = 20.0        # floor (px) for the inflated matching box
    max_frame_gap: int = 30           # frames a ball track may coast unmatched
    distance_gate_px: float = 50.0    # max center distance for a zero-IoU match


@dataclass
class StitchingConfig:
    """Post-tracking ID stitching (re-link broken identities).

    Pure post-process over the exported tracks; the BoT-SORT tracker is
    not modified. If a new id appears near where an old one disappeared
    (within the gap + predicted-position gates), it is remapped back to
    the old id.
    """

    enabled: bool = False
    max_frame_gap: int = 45          # frames a track may vanish before re-link
    distance_gate_px: float = 120.0  # max gap between predicted & new position
    size_ratio_gate: float = 1.6     # bbox size compatibility (max ratio)
    match_same_class: bool = True    # only stitch tracks of the same class
    min_segment_length: int = 8      # drop noise tracks shorter than this (ball kept)
    # Appearance (jersey-colour) gate: never stitch two different-colour
    # players into one id. OFF by default -- on this footage the colour
    # sampling is noisy enough that it un-merged correct chains and made
    # tracking worse; enable only if you see cross-team merges.
    appearance_gate: bool = False
    appearance_min_sim: float = 0.5         # min cosine colour similarity to merge


@dataclass
class SwapCorrectionConfig:
    """Post-tracking appearance-based ID-swap correction.

    Fixes the case stitching can't: two players crossing and *exchanging*
    IDs. Detected from jersey color (a track that flips team mid-way) and
    undone from the crossing frame onward.
    """

    enabled: bool = False
    samples_per_track: int = 40       # appearance samples per track (temporal)
    min_track_length: int = 30        # only consider reasonably long tracks
    frame_tol: int = 25               # max frame gap between the paired flips
    distance_gate_px: float = 220.0   # how close the two tracks must cross
    min_purity: float = 0.8           # how clean the before/after split must be
    # Split a single-track DRIFT (a flip interval with no crossing partner)
    # into its own id. OFF by default -- it relies on the same noisy colour
    # clustering and can fragment good tracks; enable only when needed.
    split_unpaired: bool = False
    min_split_samples: int = 3        # off-team samples needed to split
    min_split_purity: float = 0.7     # only split a track with a clear home team
    # Where to make the exchange. Appearance says THAT two tracks swapped but
    # not WHEN -- the jersey only reads as the other team once the players
    # have drawn apart, tens of frames later. Cutting there makes each
    # repaired track teleport across that gap; the window is instead nudged to
    # the cheapest join nearby, which is the crossing itself.
    edge_refine_frames: int = 45


@dataclass
class RosterConfig:
    """Fixed squad numbers: 1..11 one team, 12..22 the other, 0 the ball.

    Bounds the ids instead of letting the tracker mint them, so a number means
    a person for the whole clip and running out is visible. The keeper is one
    of the eleven rather than a slot of its own, because a broadcast frame
    does not reach both goals -- on Tesr1, two keepers were never both visible
    in any of 1471 frames.

    Colour separates the two kits, not the eleven inside one, so two
    team-mates can still exchange numbers unnoticed.
    """

    enabled: bool = False
    slots_per_team: int = 11
    max_gap_frames: int = 75          # a number is freed after this long unworn
    distance_gate_px: float = 250.0   # how far a returning player may reappear
    unnumbered_classes: Tuple[str, ...] = ("referee",)
    # Every Nth frame is projected to metres. Roles are read off the shape of
    # a whole track, so the grid only has to be fine enough that two tracks
    # share frames to be compared on.
    pitch_sample_every: int = 5


@dataclass
class TrackingConfig:
    """Phase 2 tracking configuration."""

    enabled: bool = False
    tracker_type: str = "botsort"     # "botsort" (preferred) | "bytetrack"
    separate_ball: bool = True        # track the ball with its own tracker
    ball_class_name: str = "ball"
    with_reid: bool = False           # opt-in; config.yaml turns it on
    reid_model: str = "auto"          # "auto" is unsupported standalone -- use a checkpoint
    # Give the tracker the boxes that failed the per-class display thresholds.
    # BoT-SORT's second association pass is over low-score boxes; without them
    # a player whose score dips during an occlusion loses their track and comes
    # back as a new id. Weak boxes are never exported or drawn.
    feed_weak_detections: bool = True
    save_tracks: bool = True
    output_path: Optional[str] = None  # override for the tracking JSON path
    # Main tracker handles player / goalkeeper / referee.
    main: BotSortParams = field(default_factory=BotSortParams)
    # The ball is small and unstable, so it gets looser, lower thresholds
    # and a longer buffer to survive frequent misses. Only used as a
    # fallback when ball_tracking.enabled is False.
    ball: BotSortParams = field(
        default_factory=lambda: BotSortParams(
            track_high_thresh=0.1,
            track_low_thresh=0.05,
            new_track_thresh=0.1,
            track_buffer=60,
            # The ball's confidence is inherently low; fusing it into the
            # match cost would break re-association, so leave it off.
            fuse_score=False,
        )
    )
    # Custom ball association (preferred over the BoT-SORT ball tracker).
    ball_tracking: BallTrackingConfig = field(default_factory=BallTrackingConfig)
    # Post-tracking ID stitching (re-link broken identities).
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    # Post-tracking appearance ID-swap correction (fix crossing swaps).
    swap_correction: SwapCorrectionConfig = field(
        default_factory=SwapCorrectionConfig)
    # Fixed squad numbers over the corrected tracks.
    roster: RosterConfig = field(default_factory=RosterConfig)


@dataclass
class RoleRefinementConfig:
    """Phase 3 — track-level role refinement.

    Adaptive, per-match role correction layered on top of tracking. No
    fixed color rules: the two main team appearance clusters are
    discovered from the tracked players, and roles are decided by
    appearance outlier analysis plus temporal voting over each track.
    """

    enabled: bool = False
    min_track_length: int = 15        # frames a track must span to be "stable"
    samples_per_track: int = 20       # frames sampled per track for appearance
    jersey_crop_top: float = 0.15     # top of jersey region (fraction of bbox h)
    jersey_crop_bottom: float = 0.55  # bottom of jersey region (fraction of bbox h)
    jersey_crop_side: float = 0.15    # trim this fraction off each side (arms/bg)
    team_cluster_count: int = 2       # number of team appearance clusters
    outlier_distance_threshold: float = 0.35  # cosine dist beyond which = outlier
    temporal_vote_threshold: float = 0.70     # min winning vote fraction to commit
    unknown_if_low_confidence: bool = True    # fall back to "unknown" below threshold
    # A long, stable, non-outlier track is certainly a player on one of the
    # two teams — assign it to its leaning team instead of "unknown" even if
    # the team vote was split (common for low-saturation white kits).
    assign_team_for_stable_players: bool = True
    stable_min_length: int = 45               # frames a track must span to qualify
    # Goalkeeper vs (assistant) referee, decided by position AND motion. An
    # outlier is a keeper if it is clearly at the goal line (edge proximity
    # >= goalkeeper_min_edge) OR barely moves (roam <= goalkeeper_max_roam);
    # a side outlier that ROAMS is the assistant referee. This keeps a still
    # keeper at a moderate edge from being mislabelled a referee.
    goalkeeper_min_edge: float = 0.70
    goalkeeper_max_roam: float = 0.30
    # Final automatic reconciliation: never leave a person track "unknown".
    # Any track refinement could not commit is force-resolved to its best
    # guess — nearest team by jersey color (majority over its sampled
    # frames), an ambiguous outlier to referee/goalkeeper by pitch position,
    # and a colorless leftover to the nearest already-labeled teammate. This
    # is what makes a fresh match come out fully labeled with no manual step.
    resolve_unknowns: bool = True
    # Detector referee prior: if the detector said "referee" but refinement
    # flipped it to "player" with only moderate confidence, trust the
    # detector and keep it a referee (assistant refs often look like a team).
    referee_detector_prior_enabled: bool = True
    referee_to_player_override_confidence: float = 0.90
    ball_class_name: str = "ball"
    save_roles: bool = True
    output_path: Optional[str] = None  # override for the roles JSON path
    save_role_video: bool = False      # render an annotated role video
    random_seed: int = 42              # determinism for clustering


@dataclass
class ManualCorrectionConfig:
    """Phase 3.5 — human-in-the-loop role correction.

    A thin review layer on top of automatic role refinement: only
    *ambiguous* tracks (role ``unknown`` or low confidence) are surfaced
    for manual review, the reviewer's choices are stored separately, and
    a final role is produced by overlaying corrections on the automatic
    results. The automatic results are never overwritten.
    """

    enabled: bool = False
    review_confidence_threshold: float = 0.65   # below this -> review candidate
    review_roles: list = field(default_factory=lambda: ["unknown"])  # always review
    representative_frames: int = 9              # frames sampled per track for review
    corrections_file: str = "outputs/manual_role_corrections.json"
    review_dir: str = "outputs/manual_review"
    save_review_frames: bool = True            # also save full frames, not just crops
    save_final_video: bool = False             # render the final annotated video
    final_output_path: Optional[str] = None    # override for the final roles JSON
    # User-guided initialization (label boxes in a few frames as seeds).
    initialization_labels_file: str = "outputs/manual_initialization_labels.json"
    autosave: bool = False                     # UI: save automatically after edits


@dataclass
class CalibrationConfig:
    """Phase 4 — manual pitch calibration + minimap."""

    calibration_path: Optional[str] = None     # pitch_calibration.json to use
    save_minimap: bool = False                 # render the top-down minimap video
    px_per_meter: float = 8.0                  # minimap resolution
    reprojection_warn_threshold: float = 5.0   # warn above this (meters)
    minimap_output: Optional[str] = None       # override minimap video path
    field_positions_output: Optional[str] = None  # override field-positions JSON
    # Automatic calibration from a YOLO-pose pitch-keypoint model (no manual
    # clicking; per-frame homography that follows the camera).
    auto: bool = False
    # Smooth the per-frame homography by propagating the keypoint anchors with
    # estimated camera motion (removes the keyframe-snapping jitter that
    # teleports minimap dots). Falls back to nearest-keyframe if it can't run.
    camera_motion: bool = True
    # Auto-calibration backend: "yolo_pose" (28-kp, default) or "nbjw" (57-kp
    # HRNet, ~0.30 m vs ~1.36 m on broadcast -- opt in with --keypoint-backend
    # nbjw). Kept opt-in so the working default is never changed silently.
    keypoint_backend: str = "yolo_pose"
    keypoint_model: str = "weights/pitch_keypoints_yolo11m_best.pt.pt"
    nbjw_model: str = "weights/keypoints_best.pt.pt"
    nbjw_kp_threshold: float = 0.1486          # heatmap-peak gate for NBJW
    keypoint_sample_every: int = 5             # calibrate every Nth frame
    keypoint_min_points: int = 5               # min keypoints to solve a frame
    keypoint_min_conf: float = 0.5             # per-keypoint confidence gate
    keypoint_max_error_m: float = 4.0          # reject frames above this error


@dataclass
class MinimapConfig:
    """Phase 4 — minimap dot coloring / labels."""

    color_mode: str = "team_role"   # "team_role" (analytics) | "track_id" (debug)
    show_track_ids: bool = False    # draw the track id inside each dot
    px_per_meter: float = 9.0       # minimap resolution
    player_dot_radius: int = 9      # bigger = more visible players
    ball_dot_radius: int = 6
    stripes: int = 12               # mown-stripe count (0 = flat green)
    # Temporal smoothing of projected dots (kills per-frame homography jitter
    # that can teleport a dot across the pitch and look like an ID switch).
    smooth_alpha: float = 0.5       # 1.0 = no smoothing, lower = smoother
    max_step_m: float = 3.0         # clamp per-frame field movement (meters)
    # Colour each player box/dot by its jersey colour in THAT frame (not by the
    # track's per-track team). This keeps the team colour correct even when a
    # track id switches players. Adds one video pass.
    team_color_per_frame: bool = True
    # Tactical minimap look: connect each team's players into a convex 'block'
    # with a dashed average-position line, and use bigger dots. Referees are
    # dropped from the minimap so only the two teams (+keepers/ball) show.
    tactical_style: bool = True
    hide_referees: bool = True


@dataclass
class MinimapOverlayConfig:
    """Phase 4 — picture-in-picture minimap inside the main video."""

    enabled: bool = False
    position: str = "top_right"   # top_right | top_left | bottom_right | bottom_left
    width_ratio: float = 0.28     # overlay width as a fraction of video width
    margin: int = 20              # px from the frame edge
    background_alpha: float = 0.65  # darkness of the backing panel
    border: bool = True           # thin white border around the minimap


@dataclass
class PoseConfig:
    """Optional cosmetic — draw player pose skeletons on the final video."""

    enabled: bool = False
    model: str = "yolo11m-pose.pt"   # pretrained COCO human-pose (auto-downloads)
    conf: float = 0.35
    min_keypoint_conf: float = 0.5
    # When the skeleton is on, draw ONLY skeletons (no plain boxes). A player
    # the pose model misses gets a small dot instead of a box (set True to
    # fall back to a full box instead).
    draw_box_fallback: bool = False


@dataclass
class PossessionConfig:
    """Phase 6 — ball possession analytics (image-space, no homography)."""

    enabled: bool = False
    gate_factor: float = 2.0     # ball within gate_factor * player height of feet
    min_hold: int = 3            # frames a new team must hold before possession flips
    show_bar: bool = True        # draw a running possession bar on the final video


@dataclass
class AnalyticsConfig:
    """Phase 5 — speed & distance analytics (uses field metres from minimap)."""

    enabled: bool = True
    smoothing_window: int = 5      # moving-average window over per-frame speed
    max_speed_kmh: float = 38.0    # above this a step is an impossible jump
    min_samples: int = 5           # fewer valid samples -> insufficient_data
    # Draw the per-player speed/distance stat bar on the final video. ON by
    # default so a normal run shows it without an extra flag.
    show_speed_overlay: bool = True
    # Roles excluded from analytics + stat bars (referees / noise tracks).
    exclude_roles: list = field(default_factory=lambda: ["referee", "unknown"])
    # Drop tracks with fewer than this many valid samples — short tracking
    # fragments that would otherwise inflate the player count past ~22.
    min_track_samples: int = 30
    # Phase 7: render team + best-player heatmaps (PNG) from field positions.
    heatmaps: bool = True
    heatmap_sigma_m: float = 2.5   # Gaussian smoothing radius (metres)
    tactical: bool = True          # Phase 8: team tactical analysis JSON
    final_report: bool = True      # Phase 10: AI match intelligence report


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    classes: Dict[int, str] = field(default_factory=dict)
    video: VideoConfig = field(default_factory=VideoConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    role_refinement: RoleRefinementConfig = field(
        default_factory=RoleRefinementConfig
    )
    manual_correction: ManualCorrectionConfig = field(
        default_factory=ManualCorrectionConfig
    )
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    minimap: MinimapConfig = field(default_factory=MinimapConfig)
    minimap_overlay: MinimapOverlayConfig = field(
        default_factory=MinimapOverlayConfig
    )
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)


def _build_section(cls, raw: dict):
    """Instantiate a config dataclass from a raw dict, ignoring unknown keys."""
    valid = {f for f in cls.__dataclass_fields__}
    kwargs = {k: v for k, v in (raw or {}).items() if k in valid}
    return cls(**kwargs)


def _build_tracking(raw: dict) -> TrackingConfig:
    """Build a TrackingConfig, including its nested per-tracker params."""
    raw = dict(raw or {})
    nested_keys = {"main", "ball", "ball_tracking", "stitching",
                   "swap_correction", "roster"}
    scalar = {k: v for k, v in raw.items() if k not in nested_keys}
    tracking = _build_section(TrackingConfig, scalar)
    tracking.main = _build_section(BotSortParams, raw.get("main"))
    tracking.stitching = _build_section(StitchingConfig, raw.get("stitching"))
    tracking.swap_correction = _build_section(
        SwapCorrectionConfig, raw.get("swap_correction"))
    tracking.roster = _build_section(RosterConfig, raw.get("roster"))
    # Ball params default to the looser preset; only override what's given.
    ball_defaults = TrackingConfig().ball
    ball_raw = raw.get("ball") or {}
    merged_ball = {
        f: ball_raw.get(f, getattr(ball_defaults, f))
        for f in BotSortParams.__dataclass_fields__
    }
    tracking.ball = BotSortParams(**merged_ball)
    tracking.ball_tracking = _build_section(
        BallTrackingConfig, raw.get("ball_tracking")
    )
    return tracking


def _validate_botsort_params(name: str, params: BotSortParams) -> None:
    bounded = {
        "track_high_thresh": params.track_high_thresh,
        "track_low_thresh": params.track_low_thresh,
        "new_track_thresh": params.new_track_thresh,
        "match_thresh": params.match_thresh,
        "proximity_thresh": params.proximity_thresh,
        "appearance_thresh": params.appearance_thresh,
    }
    for key, value in bounded.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"tracking.{name}.{key} must be in [0, 1], got {value}"
            )
    if params.track_buffer <= 0:
        raise ValueError(
            f"tracking.{name}.track_buffer must be positive, "
            f"got {params.track_buffer}"
        )


def _validate(config: AppConfig) -> None:
    if not 0.0 <= config.model.confidence_threshold <= 1.0:
        raise ValueError(
            f"confidence_threshold must be in [0, 1], "
            f"got {config.model.confidence_threshold}"
        )
    if not 0.0 <= config.model.iou_threshold <= 1.0:
        raise ValueError(
            f"iou_threshold must be in [0, 1], got {config.model.iou_threshold}"
        )
    for name, value in config.model.class_confidence_overrides.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"class confidence override for '{name}' must be in [0, 1], "
                f"got {value}"
            )
    if config.model.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {config.model.image_size}")
    if not config.classes:
        raise ValueError("config must define at least one entry under 'classes'")

    tracking = config.tracking
    if tracking.tracker_type not in {"botsort", "bytetrack"}:
        raise ValueError(
            f"tracking.tracker_type must be 'botsort' or 'bytetrack', "
            f"got '{tracking.tracker_type}'"
        )
    _validate_botsort_params("main", tracking.main)
    _validate_botsort_params("ball", tracking.ball)

    bt = tracking.ball_tracking
    if bt.matching_box_scale <= 0:
        raise ValueError(
            f"ball_tracking.matching_box_scale must be > 0, "
            f"got {bt.matching_box_scale}"
        )
    if bt.min_box_size < 0:
        raise ValueError(
            f"ball_tracking.min_box_size must be >= 0, got {bt.min_box_size}"
        )
    if bt.max_frame_gap < 0:
        raise ValueError(
            f"ball_tracking.max_frame_gap must be >= 0, got {bt.max_frame_gap}"
        )
    if bt.distance_gate_px < 0:
        raise ValueError(
            f"ball_tracking.distance_gate_px must be >= 0, "
            f"got {bt.distance_gate_px}"
        )

    _validate_role_refinement(config.role_refinement)
    _validate_manual_correction(config.manual_correction)
    _validate_minimap_overlay(config.minimap_overlay)
    if config.minimap.color_mode not in {"team_role", "track_id"}:
        raise ValueError(
            "minimap.color_mode must be 'team_role' or 'track_id', "
            f"got '{config.minimap.color_mode}'"
        )


def _validate_minimap_overlay(mo: "MinimapOverlayConfig") -> None:
    valid = {"top_right", "top_left", "bottom_right", "bottom_left"}
    if mo.position not in valid:
        raise ValueError(
            f"minimap_overlay.position must be one of {sorted(valid)}, "
            f"got '{mo.position}'"
        )
    if not 0.0 < mo.width_ratio <= 1.0:
        raise ValueError(
            f"minimap_overlay.width_ratio must be in (0, 1], got {mo.width_ratio}"
        )
    if not 0.0 <= mo.background_alpha <= 1.0:
        raise ValueError(
            "minimap_overlay.background_alpha must be in [0, 1], "
            f"got {mo.background_alpha}"
        )
    if mo.margin < 0:
        raise ValueError(
            f"minimap_overlay.margin must be >= 0, got {mo.margin}"
        )


def _validate_role_refinement(rr: "RoleRefinementConfig") -> None:
    if rr.min_track_length <= 0:
        raise ValueError(
            f"role_refinement.min_track_length must be positive, "
            f"got {rr.min_track_length}"
        )
    if rr.samples_per_track <= 0:
        raise ValueError(
            f"role_refinement.samples_per_track must be positive, "
            f"got {rr.samples_per_track}"
        )
    if not 0.0 <= rr.jersey_crop_top < rr.jersey_crop_bottom <= 1.0:
        raise ValueError(
            "role_refinement requires 0 <= jersey_crop_top < jersey_crop_bottom "
            f"<= 1, got top={rr.jersey_crop_top}, bottom={rr.jersey_crop_bottom}"
        )
    if not 0.0 <= rr.jersey_crop_side < 0.5:
        raise ValueError(
            f"role_refinement.jersey_crop_side must be in [0, 0.5), "
            f"got {rr.jersey_crop_side}"
        )
    if rr.team_cluster_count < 1:
        raise ValueError(
            f"role_refinement.team_cluster_count must be >= 1, "
            f"got {rr.team_cluster_count}"
        )
    for key in (
        "outlier_distance_threshold",
        "temporal_vote_threshold",
        "referee_to_player_override_confidence",
    ):
        value = getattr(rr, key)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"role_refinement.{key} must be in [0, 1], got {value}"
            )


def _validate_manual_correction(mc: "ManualCorrectionConfig") -> None:
    if not 0.0 <= mc.review_confidence_threshold <= 1.0:
        raise ValueError(
            "manual_correction.review_confidence_threshold must be in [0, 1], "
            f"got {mc.review_confidence_threshold}"
        )
    if mc.representative_frames < 1:
        raise ValueError(
            "manual_correction.representative_frames must be >= 1, "
            f"got {mc.representative_frames}"
        )


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the application configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    vis_raw = dict(raw.get("visualization") or {})
    # YAML lists -> tuples for colors.
    if "colors" in vis_raw and vis_raw["colors"]:
        vis_raw["colors"] = {
            name: tuple(int(c) for c in bgr)
            for name, bgr in vis_raw["colors"].items()
        }

    config = AppConfig(
        model=_build_section(ModelConfig, raw.get("model")),
        classes={int(k): str(v) for k, v in (raw.get("classes") or {}).items()},
        video=_build_section(VideoConfig, raw.get("video")),
        visualization=_build_section(VisualizationConfig, vis_raw),
        debug=_build_section(DebugConfig, raw.get("debug")),
        logging=_build_section(LoggingConfig, raw.get("logging")),
        tracking=_build_tracking(raw.get("tracking")),
        role_refinement=_build_section(
            RoleRefinementConfig, raw.get("role_refinement")
        ),
        manual_correction=_build_section(
            ManualCorrectionConfig, raw.get("manual_correction")
        ),
        calibration=_build_section(CalibrationConfig, raw.get("calibration")),
        minimap=_build_section(MinimapConfig, raw.get("minimap")),
        minimap_overlay=_build_section(
            MinimapOverlayConfig, raw.get("minimap_overlay")
        ),
        possession=_build_section(PossessionConfig, raw.get("possession")),
        pose=_build_section(PoseConfig, raw.get("pose")),
        analytics=_build_section(AnalyticsConfig, raw.get("analytics")),
    )
    _validate(config)
    return config
