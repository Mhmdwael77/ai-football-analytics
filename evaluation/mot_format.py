"""MOTChallenge-format I/O — the bridge between our tracks JSON and the
standard tracking benchmark.

MOTChallenge stores one comma-separated row per box, ten columns:

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

The last three are 3-D world coords, unused for 2-D benchmarks and always
``-1`` here. Ground truth uses ``conf`` as a "consider this box" flag (1) and
predictions use the detector score.

Two directions are supported:

* :func:`tracks_json_to_mot` — turn a pipeline output
  (``outputs/<clip>_tracks.json``) into a MOT file. This is also how you
  *start* a ground truth: export the tracker's own output, then hand-fix the
  ids in a spreadsheet or a labelling tool. Correcting is far faster than
  annotating from scratch, and the frame/box columns are already right.
* :func:`read_mot` — read either file back for scoring.

Nothing here imports ``motmetrics``; the format layer stays dependency-free
so it can be unit-tested on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from utils.logger import get_logger

logger = get_logger("evaluation.mot_format")

# Classes scored by default. The ball is excluded: it is a single object with
# no identity to confuse, so including it inflates MOTA and tells you nothing
# about ID switches. Pass an explicit class list to score it anyway.
DEFAULT_CLASSES = ("player", "goalkeeper", "referee")


@dataclass(frozen=True)
class MotRow:
    """One MOTChallenge row (boxes are stored as xywh, top-left origin)."""

    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0

    @property
    def xywh(self) -> List[float]:
        return [self.x, self.y, self.w, self.h]

    def to_line(self) -> str:
        return (
            f"{self.frame},{self.track_id},{self.x:.2f},{self.y:.2f},"
            f"{self.w:.2f},{self.h:.2f},{self.conf:.4f},-1,-1,-1"
        )


def xyxy_to_xywh(bbox: Sequence[float]) -> List[float]:
    """``[x1, y1, x2, y2]`` -> ``[left, top, width, height]``."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return [x1, y1, x2 - x1, y2 - y1]


def read_mot(path: str | Path) -> Dict[int, List[MotRow]]:
    """Read a MOT file into ``{frame -> [MotRow, ...]}``.

    Blank lines and ``#`` comments are skipped so a hand-edited ground truth
    can carry notes without breaking the parser.
    """
    rows: Dict[int, List[MotRow]] = {}
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise ValueError(
                f"{path}:{lineno}: expected at least 6 comma-separated columns, "
                f"got {len(parts)} — is this a MOTChallenge file?"
            )
        conf = float(parts[6]) if len(parts) > 6 and parts[6] not in ("", "-1") else 1.0
        row = MotRow(
            frame=int(float(parts[0])),
            track_id=int(float(parts[1])),
            x=float(parts[2]),
            y=float(parts[3]),
            w=float(parts[4]),
            h=float(parts[5]),
            conf=conf,
        )
        rows.setdefault(row.frame, []).append(row)
    return rows


def write_mot(rows: Iterable[MotRow], path: str | Path) -> Path:
    """Write rows to a MOT file, sorted by (frame, id). Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r.frame, r.track_id))
    out.write_text("\n".join(r.to_line() for r in ordered) + "\n", encoding="utf-8")
    logger.info("Wrote %d rows to %s", len(ordered), out)
    return out


def tracks_json_to_rows(
    tracks_json: str | Path,
    classes: Optional[Sequence[str]] = DEFAULT_CLASSES,
    as_ground_truth: bool = False,
) -> List[MotRow]:
    """Convert an exported tracks JSON into MOT rows.

    Args:
        tracks_json: Path to ``outputs/<clip>_tracks.json`` (or any file with
            the same ``{"frames": [{"frame": n, "tracks": [...]}]}`` schema).
        classes: Keep only these class names; ``None`` keeps every class.
        as_ground_truth: Write ``conf = 1`` on every row (MOT ground-truth
            convention) instead of the tracker's confidence.
    """
    data = json.loads(Path(tracks_json).read_text(encoding="utf-8"))
    keep = set(classes) if classes else None

    rows: List[MotRow] = []
    for frame in data.get("frames", []):
        frame_no = int(frame["frame"])
        for track in frame.get("tracks", []):
            if keep is not None and track.get("class") not in keep:
                continue
            x, y, w, h = xyxy_to_xywh(track["bbox"])
            rows.append(MotRow(
                frame=frame_no,
                track_id=int(track["track_id"]),
                x=x, y=y, w=w, h=h,
                conf=1.0 if as_ground_truth else float(track.get("confidence", 1.0)),
            ))
    logger.info("Converted %s -> %d MOT rows", Path(tracks_json).name, len(rows))
    return rows


def tracks_json_to_mot(
    tracks_json: str | Path,
    out_path: str | Path,
    classes: Optional[Sequence[str]] = DEFAULT_CLASSES,
    as_ground_truth: bool = False,
) -> Path:
    """Convert a tracks JSON straight to a MOT file on disk."""
    return write_mot(
        tracks_json_to_rows(tracks_json, classes, as_ground_truth), out_path)


def frame_range(rows: Dict[int, List[MotRow]]) -> tuple:
    """``(first_frame, last_frame)`` present in a parsed MOT file."""
    if not rows:
        return (0, 0)
    return (min(rows), max(rows))
