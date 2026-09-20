"""Tracking accuracy as MOTA / MOTP / IDF1 — and the ID-switch count.

Scores an exported tracks JSON against a hand-checked ground truth using the
standard MOTChallenge protocol (``py-motmetrics``). The headline numbers:

* **IDsw**  — identity switches. The number this project is optimising.
* **IDF1**  — how well identities are *preserved* over time. The metric that
  actually moves when ID switches drop; MOTA barely notices them.
* **MOTA**  — detection-dominated summary (FP + FN + IDsw over GT boxes).
* **MOTP**  — localisation quality of the matched boxes.

MOTA is reported because reviewers expect it, but do not tune against it: a
run can gain MOTA while getting *worse* at identity. Read IDF1 and IDsw.

Getting a ground truth
----------------------
Annotating from scratch is slow. Export the tracker's own output as a
starting point and hand-fix the ids::

    python -m evaluation.mota_eval --make-gt-template
        outputs/clasico_clip_tracks.json -o evaluation/_dataset/clasico_gt.txt

Then open the file, follow each player, and correct the id column wherever the
tracker switched. Two minutes of footage is enough to be meaningful.

Usage
-----
    # score one run
    python -m evaluation.mota_eval --gt evaluation/_dataset/clasico_gt.txt
        --pred outputs/clasico_clip_tracks.json

    # A/B two runs against the same ground truth (e.g. ReID off vs on)
    python -m evaluation.mota_eval --gt evaluation/_dataset/clasico_gt.txt
        --pred outputs/before_tracks.json --pred outputs/after_tracks.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from evaluation.mot_format import (
    DEFAULT_CLASSES,
    MotRow,
    read_mot,
    tracks_json_to_rows,
    write_mot,
)
from utils.logger import get_logger

logger = get_logger("evaluation.mota_eval")

# Metric names requested from motmetrics, in report order.
_METRICS = [
    "idf1", "idp", "idr",
    "mota", "motp",
    "precision", "recall",
    "num_switches", "num_fragmentations",
    "num_false_positives", "num_misses",
    "mostly_tracked", "partially_tracked", "mostly_lost",
    "num_unique_objects", "num_objects", "num_frames",
]


def _require_motmetrics():
    try:
        import motmetrics as mm
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "py-motmetrics is not installed. Run:\n"
            "    pip install motmetrics\n"
            "(it is listed in requirements.txt as an evaluation-only extra)."
        ) from exc
    return mm


def _load_rows(
    path: str | Path, classes: Optional[Sequence[str]]
) -> Dict[int, List[MotRow]]:
    """Load either a MOT ``.txt`` or an exported tracks ``.json``."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        rows: Dict[int, List[MotRow]] = {}
        for row in tracks_json_to_rows(p, classes):
            rows.setdefault(row.frame, []).append(row)
        return rows
    return read_mot(p)


def accumulate(
    gt: Dict[int, List[MotRow]],
    pred: Dict[int, List[MotRow]],
    iou_threshold: float = 0.5,
):
    """Build a populated ``MOTAccumulator`` over the frames present in ``gt``.

    Only ground-truth frames are scored: a sparsely annotated clip (say every
    frame of a 2-minute window) must not be punished for predictions outside
    the annotated range.
    """
    mm = _require_motmetrics()
    acc = mm.MOTAccumulator(auto_id=False)

    # motmetrics discards a pair when (1 - IoU) exceeds max_iou, so the gate
    # for "IoU >= threshold" is max_iou = 1 - threshold.
    max_dist = 1.0 - iou_threshold

    for frame in sorted(gt):
        gt_rows = gt[frame]
        pred_rows = pred.get(frame, [])
        gt_ids = [r.track_id for r in gt_rows]
        pred_ids = [r.track_id for r in pred_rows]
        dists = mm.distances.iou_matrix(
            [r.xywh for r in gt_rows],
            [r.xywh for r in pred_rows],
            max_iou=max_dist,
        )
        acc.update(gt_ids, pred_ids, dists, frameid=frame)
    return acc


def evaluate(
    gt_path: str | Path,
    pred_path: str | Path,
    iou_threshold: float = 0.5,
    classes: Optional[Sequence[str]] = DEFAULT_CLASSES,
) -> Dict:
    """Score one prediction file against the ground truth."""
    mm = _require_motmetrics()

    gt = _load_rows(gt_path, classes)
    pred = _load_rows(pred_path, classes)
    if not gt:
        raise SystemExit(f"Ground truth {gt_path} has no rows.")

    acc = accumulate(gt, pred, iou_threshold)
    host = mm.metrics.create()
    summary = host.compute(acc, metrics=_METRICS, name="run")
    values = {k: summary.loc["run", k] for k in _METRICS}

    n_frames = max(int(values["num_frames"]), 1)
    switches = int(values["num_switches"])
    return {
        "metadata": {
            "ground_truth": str(gt_path),
            "prediction": str(pred_path),
            "iou_threshold": iou_threshold,
            "classes": list(classes) if classes else "all",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "identity": {
            "IDF1": float(values["idf1"]),
            "IDP": float(values["idp"]),
            "IDR": float(values["idr"]),
            "id_switches": switches,
            "id_switches_per_1000_frames": round(switches * 1000.0 / n_frames, 2),
            "fragmentations": int(values["num_fragmentations"]),
        },
        "detection": {
            "MOTA": float(values["mota"]),
            "MOTP": float(values["motp"]),
            "precision": float(values["precision"]),
            "recall": float(values["recall"]),
            "false_positives": int(values["num_false_positives"]),
            "misses": int(values["num_misses"]),
        },
        "coverage": {
            "mostly_tracked": int(values["mostly_tracked"]),
            "partially_tracked": int(values["partially_tracked"]),
            "mostly_lost": int(values["mostly_lost"]),
            "unique_objects": int(values["num_unique_objects"]),
            "gt_boxes": int(values["num_objects"]),
            "frames_scored": n_frames,
        },
    }


def write_reports(runs: List[Dict], out_dir: Path) -> Path:
    """Write tracking metrics as JSON + Markdown; return the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = runs[0] if len(runs) == 1 else {"runs": runs}
    json_path = out_dir / "tracking_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Tracking accuracy — MOTA / IDF1 / ID switches",
        "",
        f"_Ground truth:_ `{runs[0]['metadata']['ground_truth']}`  ",
        f"_IoU threshold:_ {runs[0]['metadata']['iou_threshold']}  ",
        f"_Classes:_ {runs[0]['metadata']['classes']}  ",
        f"_Generated:_ {runs[0]['metadata']['created_at']}",
        "",
        "## Identity (the numbers that matter here)",
        "",
        "| Run | IDF1 | ID switches | IDsw / 1k frames | Fragmentations |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        i = run["identity"]
        lines.append(
            f"| `{Path(run['metadata']['prediction']).name}` | {i['IDF1']:.3f} | "
            f"{i['id_switches']} | {i['id_switches_per_1000_frames']:.2f} | "
            f"{i['fragmentations']} |"
        )
    lines += [
        "",
        "## Detection",
        "",
        "| Run | MOTA | MOTP | Precision | Recall |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        d = run["detection"]
        lines.append(
            f"| `{Path(run['metadata']['prediction']).name}` | {d['MOTA']:.3f} | "
            f"{d['MOTP']:.3f} | {d['precision']:.3f} | {d['recall']:.3f} |"
        )

    if len(runs) == 2:
        a, b = runs[0]["identity"], runs[1]["identity"]
        d_idf1 = b["IDF1"] - a["IDF1"]
        d_sw = b["id_switches"] - a["id_switches"]
        pct = (-d_sw / a["id_switches"] * 100.0) if a["id_switches"] else 0.0
        lines += [
            "",
            "## Delta (run 2 vs run 1)",
            "",
            f"- IDF1: **{d_idf1:+.3f}**",
            f"- ID switches: **{d_sw:+d}** ({pct:+.1f}%)",
        ]

    md_path = out_dir / "tracking_metrics.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s and %s", json_path.name, md_path.name)
    return json_path


def _print_summary(runs: List[Dict]) -> None:
    print("\n==============  TRACKING  MOTA / IDF1  ==============")
    for run in runs:
        i, d, c = run["identity"], run["detection"], run["coverage"]
        print(f"  {Path(run['metadata']['prediction']).name}")
        print(f"    IDF1         : {i['IDF1']:.4f}")
        print(f"    ID switches  : {i['id_switches']}  "
              f"({i['id_switches_per_1000_frames']:.2f} / 1k frames)")
        print(f"    fragments    : {i['fragmentations']}")
        print(f"    MOTA / MOTP  : {d['MOTA']:.4f} / {d['MOTP']:.4f}")
        print(f"    frames / GT  : {c['frames_scored']} / {c['gt_boxes']} boxes")
    if len(runs) == 2:
        a, b = runs[0]["identity"], runs[1]["identity"]
        d_sw = b["id_switches"] - a["id_switches"]
        pct = (-d_sw / a["id_switches"] * 100.0) if a["id_switches"] else 0.0
        print("  --- delta ---")
        print(f"    IDF1         : {b['IDF1'] - a['IDF1']:+.4f}")
        print(f"    ID switches  : {d_sw:+d}  ({pct:+.1f}%)")
    print("====================================================\n")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tracking accuracy (MOTA / MOTP / IDF1 / ID switches)")
    parser.add_argument(
        "--make-gt-template", metavar="TRACKS_JSON",
        help="Convert a tracks JSON into a MOT file to hand-correct into a "
             "ground truth, then exit")
    parser.add_argument("-o", "--output", help="Output path for the template")
    parser.add_argument("--gt", help="Ground-truth MOT file")
    parser.add_argument(
        "--pred", action="append", default=[],
        help="Prediction (tracks JSON or MOT file). Pass twice to A/B two runs")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for a match (default: 0.5)")
    parser.add_argument(
        "--classes", default=",".join(DEFAULT_CLASSES),
        help="Comma-separated classes to score, or 'all' (default: %(default)s)")
    parser.add_argument("--out-dir", default="outputs/evaluation")
    args = parser.parse_args(argv)

    classes = None if args.classes.lower() == "all" else tuple(
        c.strip() for c in args.classes.split(",") if c.strip())

    if args.make_gt_template:
        out = Path(args.output or "evaluation/_dataset/ground_truth.txt")
        rows = tracks_json_to_rows(args.make_gt_template, classes,
                                   as_ground_truth=True)
        write_mot(rows, out)
        print(f"\nGround-truth template written to {out}")
        print("Now correct the 2nd column (track id) wherever the tracker "
              "switched identity, then score against it with --gt.\n")
        return 0

    if not args.gt or not args.pred:
        parser.error("--gt and at least one --pred are required "
                     "(or use --make-gt-template)")

    runs = [evaluate(args.gt, p, args.iou, classes) for p in args.pred]
    _print_summary(runs)
    write_reports(runs, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
