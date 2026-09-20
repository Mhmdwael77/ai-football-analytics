"""Pull one short stretch out of a long YouTube video, ready for the pipeline.

Only the requested range is fetched, not the whole match: yt-dlp asks the
server for those byte ranges, so a minute out of a two-hour upload costs a
minute's worth of download.

Two things about the output matter more than they look, because the pipeline is
tuned in frames rather than in seconds:

  FRAME RATE. `tracking.roster.max_gap_frames` is 75, `MAX_PX_PER_FRAME` is 11,
  `track_buffer` is 30 -- all of them counted in frames, all of them chosen
  against footage at roughly 25 fps. YouTube commonly serves football at 50 or
  60, and at 50 those same numbers quietly mean half the time and half the
  speed limit: a track may vanish for 1.5 s instead of 3 before losing its
  squad number, and a normal sprint reads as an impossible jump. So the clip is
  resampled to 25 by default. Pass `--fps 0` to keep whatever the source has,
  and expect to retune if you do.

  RESOLUTION. Detection was measured at 1920 on the long side and the gap was
  large -- ids born mid-pitch went from 9 to 1. A 720p download throws that
  away before the model ever sees it, so 1080p is preferred and anything
  smaller is called out.

Usage:

    python tools/fetch_clip.py "<url>" --start 8:30 --end 9:30 --name match4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_timestamp(value: str) -> float:
    """Seconds from ``h:mm:ss``, ``m:ss`` or a plain number."""
    parts = value.strip().split(":")
    if not all(p.replace(".", "", 1).isdigit() for p in parts) or len(parts) > 3:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a timestamp; use 9:30, 1:09:30 or 570")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def as_clock(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        sys.exit(f"{tool} is not installed or not on PATH.\n"
                 f"  yt-dlp:  pip install -U yt-dlp\n"
                 f"  ffmpeg:  winget install Gyan.FFmpeg")
    return path


def probe(video: Path) -> dict:
    """Width, height, frame rate and duration, straight from the file."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", str(video)],
        capture_output=True, text=True, check=True).stdout
    info = dict(line.split("=", 1) for line in out.strip().splitlines())
    num, _, den = info.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    return {"width": int(info.get("width", 0)),
            "height": int(info.get("height", 0)),
            "fps": fps,
            "duration": float(info.get("duration", 0.0))}


def js_runtime() -> "str | None":
    """A JavaScript engine yt-dlp can borrow, if one is installed.

    YouTube signs its media URLs with code it ships as JavaScript, and yt-dlp
    has to run that code to build a working link. Without an engine the
    download reaches the server and comes back 403 Forbidden -- which reads
    like a blocked video rather than a missing dependency, so it is worth
    naming. yt-dlp only looks for deno on its own; node is far more likely to
    already be here, and does the job just as well.
    """
    for runtime in ("deno", "node", "bun"):
        if shutil.which(runtime):
            return runtime
    return None


def download(url: str, start: float, end: float, target: Path,
             max_height: int, with_audio: bool, exact: bool = False) -> None:
    """Fetch just ``[start, end]`` into ``target``.

    H.264 is asked for by name and only then anything else. Left to itself
    yt-dlp picks whichever 1080p stream is smallest, which on this upload was
    AV1 -- a third of the bytes and a codec many OpenCV builds cannot open at
    all. Every clip this pipeline reads today is H.264, and a video that will
    not open is a worse outcome than a larger download.
    """
    h264 = f"bv*[height<={max_height}][vcodec^=avc1]"
    other = f"bv*[height<={max_height}]"
    video = f"{h264}/{other}"
    fmt = (f"{h264}+ba/{other}+ba/b[height<={max_height}]" if with_audio
           else f"{video}/b")
    cmd = [
        require("yt-dlp"),
        "--download-sections", f"*{as_clock(start)}-{as_clock(end)}",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", str(target.with_suffix(".%(ext)s")),
    ]
    if exact:
        # Cuts at the exact second by re-encoding. On a long upload served over
        # HLS this crawls -- it stalled for ten minutes here without writing a
        # byte -- so it is opt-in. Without it the cut lands on the nearest
        # keyframe, a second or two out, which no test clip cares about.
        cmd.append("--force-keyframes-at-cuts")
    runtime = js_runtime()
    if runtime:
        cmd += ["--js-runtimes", runtime]
    else:
        print("  no JavaScript runtime found (deno/node/bun). If this fails\n"
              "  with 403 Forbidden, that is why: install one, or run\n"
              "  `python -m pip install -U yt-dlp` if it is simply out of date.\n")
    cmd.append(url)
    print("  " + " ".join(cmd[1:]) + "\n")
    subprocess.run(cmd, check=True)


def resample(source: Path, target: Path, fps: float) -> None:
    """Re-encode to a fixed frame rate, leaving resolution alone."""
    subprocess.run(
        [require("ffmpeg"), "-y", "-loglevel", "error", "-stats",
         "-i", str(source), "-r", str(fps),
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-an", str(target)],
        check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download one stretch of a YouTube video for the pipeline.")
    ap.add_argument("url")
    ap.add_argument("--start", type=parse_timestamp, required=True,
                    help="where the clip begins, e.g. 8:30")
    ap.add_argument("--end", type=parse_timestamp, required=True,
                    help="where it ends, e.g. 9:30")
    ap.add_argument("--name", default="clip",
                    help="output stem; the pipeline names everything after it")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="frame rate to normalise to; 0 keeps the source's")
    ap.add_argument("--max-height", type=int, default=1080,
                    help="tallest format to accept from YouTube")
    ap.add_argument("--audio", action="store_true",
                    help="keep the commentary (the pipeline ignores it)")
    ap.add_argument("--exact", action="store_true",
                    help="cut at the exact second instead of the nearest "
                         "keyframe; re-encodes, and can be very slow")
    ap.add_argument("--out-dir", type=Path, default=Path("."),
                    help="where to put the clip (default: this folder)")
    args = ap.parse_args()

    if args.end <= args.start:
        sys.exit(f"--end ({as_clock(args.end)}) must come after "
                 f"--start ({as_clock(args.start)})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    final = args.out_dir / f"{args.name}.mp4"
    if final.exists():
        sys.exit(f"{final} already exists; pick another --name or move it")

    raw = args.out_dir / f"{args.name}__raw"
    wanted = args.end - args.start
    print(f"fetching {as_clock(args.start)}..{as_clock(args.end)} "
          f"({wanted:.0f} s) from the source\n")
    download(args.url, args.start, args.end, raw, args.max_height,
             args.audio, args.exact)

    fetched = next((p for p in args.out_dir.glob(f"{args.name}__raw.*")), None)
    if fetched is None:
        sys.exit("yt-dlp finished but produced no file")
    info = probe(fetched)
    print(f"\ndownloaded : {info['width']}x{info['height']} @ "
          f"{info['fps']:.2f} fps, {info['duration']:.1f} s")

    if args.fps and abs(info["fps"] - args.fps) > 0.5:
        print(f"resampling {info['fps']:.0f} -> {args.fps:.0f} fps "
              f"(the pipeline's gates are counted in frames)")
        resample(fetched, final, args.fps)
        fetched.unlink()
    else:
        fetched.rename(final)

    out = probe(final)
    print(f"\nwrote {final}")
    print(f"  {out['width']}x{out['height']} @ {out['fps']:.2f} fps, "
          f"{out['duration']:.1f} s")

    if out["height"] < 1080:
        print(f"  NOTE: {out['height']}p. Detection was measured at 1920 on the "
              f"long side and the difference was large (ids born mid-pitch, "
              f"9 -> 1). Expect worse tracking than Tesr1.")
    if out["duration"] < wanted - 2:
        print(f"  NOTE: {out['duration']:.0f} s arrived, {wanted:.0f} s asked "
              f"for. The source may be shorter than --end.")

    print(f"\nnext: set the pitch anchors for this clip, then\n"
          f"  python main.py --video {final.name} --all")


if __name__ == "__main__":
    main()
