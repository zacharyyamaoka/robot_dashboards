"""Stills (and the dual-arm hero videos) of every build.py recording in the real Rerun web viewer, headless.

    env -u VIRTUAL_ENV uv run --directory /home/bam/robot_dashboards python docs/proposals/robot-viewer-2026-11-28/shoot.py [--video]

Reuses robot_msg_viz's tools/shoot.cjs (stills) and tools/hero.cjs (video): each starts its own `rerun --serve-web`
and headless Chrome on free ports and stops them when it ends; nothing is fronted. Stills land in
reports/media/robot-viewer-2026-11-28/shots/, videos in .../video/ (mp4 + webp poster).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MEDIA = Path("/home/bam/robot_dashboards/reports/media/robot-viewer-2026-11-28")
TOOLS = Path("/home/bam/robot_msg_viz/tools")
RERUN = "/home/bam/robot_dashboards/.venv/bin/rerun"
WIDTH, HEIGHT = 1440, 900
STILL_AT_S = "end"  # t = 5.0 s: after the right elbow's fault (2.6 s) and wrist_3 crossing 60 °C (3.7 s)
# WHY "end": a numeric seek did not land in shoot.cjs here (the stills still read t = 5.00 s), so say what they show


def stills(only: list[str]) -> int:
    rrds = sorted((MEDIA / "rrd").glob("*.rrd")) + sorted((MEDIA / "rrd" / "tabs").glob("*.rrd"))
    rrds = [r for r in rrds if not only or r.stem in only]
    jobs = [{"rrd": str(r), "out": str(MEDIA / "shots" / f"{r.stem}.png"), "time": STILL_AT_S} for r in rrds]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(jobs, handle)
    env = {**os.environ, "RERUN_BIN": RERUN, "SHOOT_WIDTH": str(WIDTH), "SHOOT_HEIGHT": str(HEIGHT),
           "SHOOT_WAIT_MS": "6000"}  # fmt: skip
    run = subprocess.run(["node", str(TOOLS / "shoot.cjs"), handle.name], capture_output=True, text=True, env=env)
    bad = 0
    for line in run.stdout.splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            ok = row["ok"] and not row["errors"]
            bad += not ok
            print(("OK  " if ok else "BAD ") + Path(row["out"]).name, "; ".join(row["errors"])[:200])
    if run.returncode:
        print(run.stderr, file=sys.stderr)
    return run.returncode or bad


def video(variant: str, clicks: list | None = None) -> None:
    """~9 s of the dual-arm cell played 0 → 5 s (and clicks, e.g. a tab), cut and encoded like hero_video.py."""
    out = MEDIA / "video" / f"{variant}-dual_arm.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        plan = Path(scratch) / "plan.json"
        plan.write_text(json.dumps({"video_dir": scratch, "width": WIDTH, "height": HEIGHT, "segments": [
            {"rrd": str(MEDIA / "rrd" / f"{variant}-dual_arm.rrd"), "seconds": 10, "play": [0, 5.0],
             "clicks": clicks or []}]}))  # fmt: skip
        run = subprocess.run(["node", str(TOOLS / "hero.cjs"), str(plan)], capture_output=True, text=True,
                             env={**os.environ, "RERUN_BIN": RERUN}, check=True)  # fmt: skip
        result = json.loads(run.stdout.strip().splitlines()[-1])
        mark = result["segments"][0]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", result["video"], "-ss", f"{mark['start']:.3f}",
                        "-to", f"{mark['end']:.3f}", "-vf", "fps=25,format=yuv420p", "-c:v", "libx264",
                        "-threads", "2", "-crf", "30", "-an", "-movflags", "+faststart", str(out)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "4.3", "-i", str(out), "-frames:v", "1",
                        "-c:v", "libwebp", "-quality", "82", str(out.with_suffix(".webp"))], check=True)
    print(f"{out}  {out.stat().st_size / 1e6:.2f} MB  errors={mark['errors']}")


# Tab clicks during the video, (at_s, x, y) in the 1440x900 viewport, measured from the stills
CLICKS: dict[str, list] = {
    "v1": [[3.5, 668, 78], [6.5, 655, 50]],  # Servos › right_arm, then Cameras
    "v2": [[3.5, 664, 50], [6.5, 786, 50]],  # right_arm tab, then overhead_camera
    "v4": [[3.5, 802, 50], [6.5, 880, 50]],  # Health, then Cameras
}

if __name__ == "__main__":
    if "--video" in sys.argv:
        for v in ("v1", "v2", "v3", "v4", "v5"):
            video(v, CLICKS.get(v))
        sys.exit(0)
    sys.exit(stills([a for a in sys.argv[1:] if not a.startswith("--")]))
