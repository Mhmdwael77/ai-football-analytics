<div align="center">

# ⚽ AI Football Analytics

### An end-to-end computer-vision system that turns raw match footage into a full tactical & performance report

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00B0FF?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/tests-340%2B%20passing-success)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<em>Detection → Tracking → Dynamic teams & roles → Pitch calibration → Speed &amp; distance → Tactics, possession &amp; an AI match report</em>

<br/>

<img src="thesis_figures/12_final_composite.png" alt="Full broadcast composite: skeleton overlays, player names, tactical minimap, possession and live stat bars" width="90%"/>

<sub><b>One command, any match.</b> Skeleton overlays · player names · tactical minimap · live speed/distance bars · possession — all discovered from the footage, nothing hard-coded.</sub>

</div>

---

## 📖 Overview

**AI Football Analytics** is a graduation project that processes a football
video from raw frames all the way to an analyst-grade report. It detects and
tracks every player, goalkeeper, referee and the ball, discovers the two teams
from their kit colours, projects the action onto a real-metre pitch, and then
computes speed, distance, heatmaps, possession, tactical profiles and a
per-player rating — finishing with a natural-language match report.

> **Design principle — nothing is hard-coded.** Teams, jersey colours, pitch
> geometry and player roles are all *learned per match* from the footage. The
> same command runs on any match, with no per-video constants.

```bash
python main.py --video match.mp4 --all --device cuda:0
```

That single command runs the **entire** 12-stage pipeline.

---

## ✨ Highlights

- 🎯 **Custom-trained YOLO11m detector** for 4 classes (player · goalkeeper · referee · ball).
- 🔗 **Identity-stable tracking** — BoT-SORT for people + a *dedicated ball tracker* (the ball is too small & fast for IoU tracking), plus appearance-based **stitching** and **swap-correction** to repair broken/swapped IDs.
- 🎽 **Dynamic team & role discovery** — teams are clustered from jersey histograms (numpy-only spherical k-means); goalkeepers/referees split by position & motion. No fixed colours.
- 🗺️ **Automatic pitch calibration** — our own trained keypoint model solves the per-frame **homography** (no manual clicking), smoothed with camera-motion estimation, projecting every player to real **pitch metres**.
- 🏃 **Speed & distance** with smoothing and impossible-jump rejection, shown live under each player.
- 🧠 **Explainable analytics** — per-player performance & 0–10 rating, team tactical profiles, heatmaps, ball possession, and a final **match report**. Every label carries the metric & reason behind it — **no external LLM anywhere**.
- 🖐️ **Human-in-the-loop (optional)** — exactly two reproducible manual touch-points (team multi-select + per-track correction with player names), overlaid on the automatic result so it's never overwritten.
- 🧪 **340+ tests** across 39 modules — runs with no GPU and no weights (fake YOLO model + fake tracker).

---

## 🎬 Pipeline in pictures

Every figure below is rendered from a **real match** processed by the system.

| Detection | Tracking | Teams & roles |
|:--:|:--:|:--:|
| ![Detection](thesis_figures/01_detection.png) | ![Tracking](thesis_figures/02_tracking.png) | ![Teams & roles](thesis_figures/03_team_role_assignment.png) |
| Raw YOLO boxes for the 4 classes | Stable BoT-SORT identities | Dynamic team clustering + GK/ref split + names |

| Pitch calibration | Skeleton pose | Tactical minimap |
|:--:|:--:|:--:|
| ![Homography](thesis_figures/05_pitch_keypoints_homography.png) | ![Pose](thesis_figures/06_skeleton_pose.png) | ![Minimap](thesis_figures/04_minimap_tactical.png) |
| Keypoints + canonical pitch reprojected | Per-player body-pose overlay | Top-down team shapes & average lines |

| Team heatmap | Speed & distance | Final composite |
|:--:|:--:|:--:|
| ![Heatmap](thesis_figures/07_heatmap_team0.png) | ![Speed/distance](thesis_figures/11_speed_distance_bar.png) | ![Composite](thesis_figures/12_final_composite.png) |
| Gaussian positional heatmap per team | Live per-player speed + distance | Full broadcast-style output |

---

## 🏗️ Architecture

A modular, layered system. Each stage is an **independent package** that talks to
the next only through **typed data contracts** (dataclasses) and JSON on disk —
never by passing raw frames around. Any stage can be re-run on a previous
stage's output.

```
            video
              │
   ┌──────────▼───────────┐
   │ detection            │  YOLO11m → typed Detections
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │ tracking             │  BoT-SORT + ball tracker → Tracks (stable ids)
   │   stitch / swap-fix  │  appearance-based identity repair
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │ role_refinement      │  dynamic team clustering → roles & teams
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │ manual_correction    │  human-in-the-loop overlay (optional)
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │ calibration          │  keypoint model → homography → field metres
   │   + minimap          │  + camera-motion smoothing, top-down render
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │ analytics            │  speed/distance, performance, heatmaps,
   │                      │  tactics, possession, match report
   └──────────────────────┘

   utils  ── config, logging, video I/O, JSON export (shared by all)
```

📚 Deep dives: **[Architecture](docs/ARCHITECTURE.md)** · **[Full pipeline](docs/PIPELINE.md)** · **[Manual workflow](docs/WORKFLOW.md)**

---

## 🔬 The 12 stages

| # | Stage | What it produces |
|---|-------|------------------|
| 1 | **Detection** | Player / goalkeeper / referee / ball boxes per frame |
| 2 | **Tracking** | Stable IDs (BoT-SORT + ball tracker, stitch + swap-fix) |
| 3 | **Role refinement** | Dynamically discovered teams + GK/referee split |
| 4 | **Manual correction** *(optional)* | Team multi-select + per-track names |
| 5 | **Homography / field projection** | Per-frame player positions in pitch metres |
| 6 | **Minimap & final video** | Tactical top-down map + skeleton broadcast video |
| 7 | **Speed & distance** | Distance covered, avg/max speed per player |
| 8 | **Player performance** | Work rate, fatigue, 0–10 rating + insight |
| 9 | **Heatmaps** | Per-team & best-player positional heatmaps |
| 10 | **Team tactical analysis** | Compactness, press, build-up, attacking zone |
| 11 | **Ball possession** | Field-space possession %, timeline, dominant team |
| 12 | **Match report** | Man of the match, key insights, recommendations |

Every analytics artifact is emitted as both machine-readable **JSON** and
human-readable **TXT**.

---

## 🚀 Getting started

### 1. Clone & install

```bash
git clone https://github.com/Mhmdwael77/ai-football-analytics.git
cd ai-football-analytics
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

### 2. Get the model weights

The trained checkpoints are published on the
**[Releases](../../releases)** page (too large for git). Download them into
`weights/`:

```bash
gh release download v1.0.0 --dir weights/
```

| File | Purpose |
|------|---------|
| `weights/best.pt` | YOLO11m detector |
| `weights/pitch_keypoints_yolo11m_best.pt.pt` | Pitch-keypoint / homography model |

See **[weights/README.md](weights/README.md)** for details. The pose model
`yolo11m-pose.pt` is fetched automatically on first use.

### 3. Run it

```bash
# Full pipeline — detection → tracking → roles → calibration → analytics → report
python main.py --video match.mp4 --all --device cuda:0

# Just detection
python main.py --video match.mp4

# Detection + tracking, with a live preview window
python main.py --video match.mp4 --tracking --display

# Quick validation on the first N frames
python main.py --video match.mp4 --all --max-frames 200
```

Outputs land in `outputs/` (annotated videos, minimap, JSON/TXT reports,
heatmaps). All tunables live in **`config/config.yaml`**.

---

## 📊 Sample output

Real match report produced by the pipeline (see
**[docs/sample_outputs/](docs/sample_outputs/)** for the full set):

```text
MATCH REPORT — input_video
========================================================

Team 0 dominated the match with 59% possession. The standout performer was
Player 21 (Team 1), rated 9.5/10. Tactically: Team 0 counter-attacking,
Team 1 attacking.

Dominant team   : Team 0
Man of the match: Player 21
Weakest player  : Player 37

TEAMS
  Team 0: Counter-Attacking | Mid Block | Compact | Fast transitions
  Team 1: Attacking | High Press | Compact | Fast transitions

KEY INSIGHTS
  - Team 0 controlled possession (59%).
  - Player 21 (Team 1) was the standout at 9.5/10 — Very active and
    frequently involved in play. Covered a lot of ground.
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

**340+ tests across 39 modules.** No model weights or GPU required — the
detector tests use a fake YOLO model, pipeline tests inject a fake tracker, and
the BoT-SORT integration tests run the real tracker on synthetic detections.

---

## 🧰 Tech stack

**Python 3.12** · **Ultralytics YOLO11** (detection · pose · BoT-SORT) ·
**OpenCV** · **NumPy** · **PyYAML** · **pytest** · **Streamlit** (optional
manual-correction UI).

---

## 📁 Project structure

```
detection/          YOLO detector + Detection contracts
tracking/           BoT-SORT, ball tracker, stitching, swap-correction
role_refinement/    dynamic team clustering + role assignment
manual_correction/  optional human-in-the-loop tools (Tk + Streamlit)
calibration/        pitch keypoints, homography, camera motion, minimap
analytics/          speed/distance, performance, heatmaps, tactics,
                    possession, match report
visualization/      box annotator + skeleton pose overlay
utils/              config, logging, video I/O, JSON export
evaluation/         detection mAP evaluation
tools/              thesis-figure & final-render generators
tests/              340+ pytest suite (no GPU / no weights needed)
config/config.yaml  every tunable in one place
docs/               architecture, pipeline & workflow docs
```

---

## 🗺️ Roadmap

All six project phases are complete:

- [x] **Phase 1** — Detection (YOLO11m)
- [x] **Phase 2** — Tracking (BoT-SORT + ball tracker, stitch & swap-fix)
- [x] **Phase 3** — Dynamic teams & role refinement (+ manual correction)
- [x] **Phase 4** — Pitch calibration & tactical minimap
- [x] **Phase 5** — Speed & distance
- [x] **Phase 6** — Player performance, tactics, possession & match report

---

## 📜 License

Released under the **[MIT License](LICENSE)**.

Copyright © 2026 Mohamed Wael. All Rights Reserved.

This project is proprietary software. No permission is granted to copy,
modify, distribute, publish, sell, sublicense, deploy, or use this software
without prior written authorization from the copyright owner.

See the [LICENSE](LICENSE) file for full terms.

---

<div align="center">
<sub>Built as a graduation project — computer vision for the beautiful game. ⚽</sub>
</div>
